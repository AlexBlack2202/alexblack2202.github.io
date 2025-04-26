# Import thư viện cần thiết
import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Khởi tạo hệ thống logging
def initLogging(logFile="train.log"):
    logging.basicConfig(
        filename=logFile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("🚀 Bắt đầu ghi log cho quá trình huấn luyện...")

# Hàm lưu best params ra file JSON
def saveBestParams(params, filePath):
    try:
        with open(filePath, "w") as f:
            json.dump(params, f, indent=4)
        logging.info(f"📝 Best Params đã lưu tại {filePath}")
    except Exception as e:
        logging.error(f"❌ Lỗi khi lưu best params: {e}")

# Hàm load best params nếu có
def loadBestParams(filePath):
    try:
        if os.path.exists(filePath):
            with open(filePath, "r") as f:
                params = json.load(f)
            logging.info(f"✅ Load Best Params từ {filePath}")
            return params
    except Exception as e:
        logging.error(f"❌ Lỗi khi load best params: {e}")
    return None

def trainSingleModel(X_train, y_train, X_test, y_test, modelName, basePipeline, paramGrid, modelDirectory):
    """
    Huấn luyện 1 mô hình duy nhất theo tên modelName, pipeline, paramGrid
    """
    print(f"🔵 Bắt đầu huấn luyện mô hình {modelName}...")
    bestScore = -np.inf
    noImproveRounds = 0
    earlyStoppingRounds = 5
    bestModel = None

    # File lưu best params
    bestParamsPath = os.path.join(modelDirectory, f"{modelName}_best_params.json")

    # Check nếu đã có best_params thì không cần GridSearch nữa
    loadedParams = loadBestParams(bestParamsPath)
    if loadedParams:
        try:
            bestModel = basePipeline.set_params(**loadedParams)
            bestModel.fit(X_train, y_train)
            print(f"✅ Đã load tham số tối ưu, không cần train lại {modelName}.")
        except Exception as e:
            logging.error(f"❌ Lỗi khi load và fit mô hình từ best params: {e}")
            loadedParams = None

    if not loadedParams:
        paramList = list(ParameterGrid(paramGrid))

        with tqdm(total=len(paramList), desc=f"Training {modelName} + EarlyStopping") as pbar:
            for params in paramList:
                try:
                    tempPipeline = basePipeline.set_params(**params)
                    scores = cross_val_score(tempPipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
                    meanScore = scores.mean()

                    if meanScore > bestScore:
                        bestScore = meanScore
                        bestParams = params
                        noImproveRounds = 0

                        # Fit mô hình tốt nhất luôn
                        bestModel = tempPipeline.fit(X_train, y_train)
                        # Save checkpoint
                        checkpointPath = os.path.join(modelDirectory, f"{modelName}_checkpoint.pkl")
                        joblib.dump(bestModel, checkpointPath)
                        logging.info(f"💾 Cập nhật checkpoint {modelName} tại {checkpointPath}")
                    else:
                        noImproveRounds += 1

                    pbar.set_postfix({"Best Score": f"{bestScore:.4f}"})
                    pbar.update(1)

                    if noImproveRounds >= earlyStoppingRounds:
                        print(f"⏹️ EarlyStopping cho {modelName} sau {earlyStoppingRounds} lần không cải thiện.")
                        break
                except Exception as e:
                    logging.error(f"❌ Lỗi khi huấn luyện với tham số {params}: {e}")
                    continue

        # Save best params ra file
        if bestParams:
            saveBestParams(bestParams, bestParamsPath)

    # Đánh giá mô hình
    if bestModel:
        yPred = bestModel.predict(X_test)
        acc = accuracy_score(y_test, yPred)
        precision = precision_score(y_test, yPred, average='weighted')
        recall = recall_score(y_test, yPred, average='weighted')
        f1 = f1_score(y_test, yPred, average='weighted')

        print(f"🎯 {modelName} - Accuracy trên tập test: {acc * 100:.2f}%")
        print(f"🎯 {modelName} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        logging.info(f"🎯 {modelName} - Test Metrics: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Lưu final model
        finalModelPath = os.path.join(modelDirectory, f"{modelName}_final.pkl")
        joblib.dump(bestModel, finalModelPath)
        logging.info(f"💾 Mô hình {modelName} đã lưu tại {finalModelPath}")

def trainEnsemblePhishingModels():
    """
    Hàm chính để huấn luyện nhiều mô hình ensemble.
    """

    # Bước 1: Setup thư mục + logging
    modelDirectory = "model"
    os.makedirs(modelDirectory, exist_ok=True)
    initLogging(os.path.join(modelDirectory, "train.log"))

    # Bước 2: Load dữ liệu
    try:
        df = pd.read_csv("data/Phishing_Email.csv")
        if "Email Text" not in df.columns or "Email Type" not in df.columns:
            raise ValueError("Dữ liệu không chứa các cột cần thiết.")
        X = df["Email Text"].fillna("")
        y = df["Email Type"]
    except Exception as e:
        logging.error(f"❌ Lỗi khi load dữ liệu: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Bước 3: Định nghĩa các pipelines
    pipelines = {
        "LogisticRegressionModel": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
        ]),
        "RandomForestModel": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))
        ])
    }

    # Bước 4: Các tập tham số tương ứng
    paramGrids = {
        "LogisticRegressionModel": {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_df": [0.8, 1.0],
            "clf__C": [0.1, 1.0, 10],
            "clf__class_weight": [None, "balanced"]
        },
        "RandomForestModel": {
            "tfidf__max_features": [None, 10000],
            "tfidf__ngram_range": [(1,1)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }
    }

    # Bước 5: Huấn luyện từng model
    for modelName, pipeline in pipelines.items():
        paramGrid = paramGrids[modelName]
        trainSingleModel(X_train, y_train, X_test, y_test, modelName, pipeline, paramGrid, modelDirectory)

if __name__ == "__main__":
    trainEnsemblePhishingModels()