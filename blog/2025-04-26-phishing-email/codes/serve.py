# Import các thư viện cần thiết
import asyncio
import json
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Bước 1: Load mô hình đã huấn luyện (đồng bộ)
model = joblib.load("model/phishingModel.pkl")  # Đảm bảo đường dẫn đúng

# Bước 2: Khởi tạo FastAPI app
app = FastAPI()

# Bước 3: Định nghĩa lớp dữ liệu vào (request) và dữ liệu ra (response)
class PredictionRequest(BaseModel):
    text: str  # Văn bản email cần phân loại

class PredictionResponse(BaseModel):
    prediction: str  # Kết quả phân loại (ví dụ: "Phishing" hoặc "Legit")
    probability: float  # Xác suất dự đoán cao nhất

# Bước 4: Định nghĩa API endpoint cho việc dự đoán
@app.post("/predict", response_model=PredictionResponse)
async def predictEmail(data: PredictionRequest):
    """
    Nhận văn bản email từ client, chạy mô hình dự đoán,
    và trả về kết quả cùng xác suất dự đoán.
    """

    # Chạy dự đoán mô hình trong thread riêng để không block event loop
    prediction = await asyncio.to_thread(model.predict, [data.text])
    probability = await asyncio.to_thread(lambda: model.predict_proba([data.text])[0].max())

    # Đóng gói kết quả trả về
    result = {
        "prediction": str(prediction[0]),
        "probability": float(probability)
    }

    return result

# Bước 5: Chạy server FastAPI với uvicorn khi file được thực thi trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# python3 serve.py


# curl 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text":"Hi There,  Watch your customer engagement soar with custom, branded messaging experience powered by artificial intelligence. Improve customer retention, conversion and satisfaction with Sendbird’s award winning communications platform.  Schedule a Custom Demo at your convenience.  Cheers, Natalie"}'

# curl 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
#   "text": "Hi, We noticed a login from a device you don'\''t usually use: android device Lagos, Nigeria If this was you, you can safely disregard this email."
# }'

# curl 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text":"STARTING AT $40 ($80)  (50% OFF APPLIED FOR LIMITED TIME)  If you are looking to learn in-demand skills and apply them directly at your workplace then Premium Courses by Great Learning Academy are a great way to get started!     Learn industry-relevant skills in Data Science and AI through a combination of expert-led hands-on projects, interactive exercises, and advanced AI support tools. These courses empower you to apply your skills effectively at work and grow in your current role or take on new projects with confidence."}'


# curl 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text":"I spend most of my time building and improving Retrieval Augmented Generation (RAG) apps.  I trust RAGs are perhaps the most popular application of AI. It’s everywhere, from chatbots to document summaries.  I also believe that most of these apps ultimately go undeployed for various reasons, many of which are not technical. However, I wish I had known a few technical aspects to create more effective RAGs."}'