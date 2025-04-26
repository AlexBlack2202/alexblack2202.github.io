# Import các thư viện cần thiết
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def trainPhishingEmailModel():
    """
    Hàm thực hiện:
    - Đọc dữ liệu email từ file CSV
    - Tiền xử lý, chia dữ liệu train/test
    - Xây dựng pipeline (TF-IDF + Logistic Regression)
    - Huấn luyện mô hình
    - Đánh giá mô hình trên tập test
    - Lưu mô hình đã train vào thư mục model/
    """

    # Bước 1: Đọc dữ liệu từ file CSV
    dataFrame = pd.read_csv("data/Phishing_Email.csv")  # Cần đảm bảo đường dẫn file chính xác

    # Bước 2: Chuẩn bị dữ liệu đầu vào (features) và nhãn (labels)
    emailTexts = dataFrame["Email Text"].fillna("")  # Thay thế giá trị null bằng chuỗi rỗng
    emailLabels = dataFrame["Email Type"]            # Label: loại email (ví dụ: phishing hoặc legit)

    # Bước 3: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    emailTextsTrain, emailTextsTest, emailLabelsTrain, emailLabelsTest = train_test_split(
        emailTexts,
        emailLabels,
        test_size=0.2,     # 20% dữ liệu dùng để kiểm tra
        random_state=42    # Đảm bảo chia dữ liệu ngẫu nhiên nhưng có thể tái lập
    )

    # Bước 4: Xây dựng pipeline:
    # - TfidfVectorizer: chuyển văn bản thành vector đặc trưng
    # - LogisticRegression: mô hình phân loại tuyến tính
    phishingDetectionPipeline = Pipeline([
        ("tfidfVectorizer", TfidfVectorizer(stop_words="english")),       # Loại bỏ từ dừng tiếng Anh
        ("logisticClassifier", LogisticRegression(solver="liblinear"))    # Sử dụng solver phù hợp với tập nhỏ
    ])

    # Bước 5: Huấn luyện pipeline trên tập huấn luyện
    phishingDetectionPipeline.fit(emailTextsTrain, emailLabelsTrain)

    # Bước 6: Đánh giá mô hình trên tập kiểm tra
    emailLabelsPredicted = phishingDetectionPipeline.predict(emailTextsTest)
    accuracy = accuracy_score(emailLabelsTest, emailLabelsPredicted)
    print(f"🎯 Độ chính xác của mô hình trên tập kiểm tra: {accuracy * 100:.2f}%")

    # Bước 7: Tạo thư mục lưu mô hình nếu chưa tồn tại
    modelDirectory = "model"
    if not os.path.exists(modelDirectory):
        os.makedirs(modelDirectory)

    # Bước 8: Lưu mô hình đã huấn luyện vào thư mục model/
    modelPath = os.path.join(modelDirectory, "phishingModel.pkl")
    joblib.dump(phishingDetectionPipeline, modelPath)
    print(f"✅ Mô hình đã được huấn luyện và lưu thành công tại '{modelPath}'.")

# Điểm bắt đầu chương trình
if __name__ == "__main__":
    trainPhishingEmailModel()
