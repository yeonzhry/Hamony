from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import os
import mysql.connector
import datetime

# --- DB 설정 (보안 주의) ---
DB_CONFIG = {
    'user': 'admin', 
    'password': 'Zmt7rwtt64',
    'host': 'database-1.cv42kwy26xf8.ap-northeast-2.rds.amazonaws.com', 
    'database': 'fastapi-ca'      
}

CURRENT_UserId = "kyung"

# --- 앱 설정 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [중요] 모델 로드 (호환성 에러 해결 코드) ---
# 최신 TF 버전에서 구버전 모델 로드 시 발생하는 'groups' 에러를 무시하는 클래스
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None) # groups 인자가 있으면 제거
        super().__init__(**kwargs)

try:
    # 커스텀 객체로 로드 시도
    model = tf.keras.models.load_model("keras_model.h5", 
                                       custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
                                       compile=False)
except Exception as e:
    print(f"Custom load failed, trying default: {e}")
    # 실패 시 기본 로드 시도
    model = tf.keras.models.load_model("keras_model.h5", compile=False)

labels = ["Re", "Mi", "Fa", "Sol", "La", "Ti", "Do"]

# 손 감지기 설정
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# --- API 라우트 ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 손 감지
        hands, _ = detector.findHands(img_cv)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # 이미지 전처리 (Crop & Resize)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Crop 범위 안전 장치
            y1, y2 = max(0, y-offset), min(img_cv.shape[0], y+h+offset)
            x1, x2 = max(0, x-offset), min(img_cv.shape[1], x+w+offset)
            
            imgCrop = img_cv[y1:y2, x1:x2]
            
            if imgCrop.size == 0:
                return {"prediction": [{"label": "No Hand", "confidence": 0.0}]}

            # 비율 맞춰 리사이징
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                try: imgWhite[:, wGap:wCal+wGap] = imgResize
                except: imgWhite = cv2.resize(imgCrop, (imgSize, imgSize))
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                try: imgWhite[hGap:hCal + hGap, :] = imgResize
                except: imgWhite = cv2.resize(imgCrop, (imgSize, imgSize))
            
            # 모델 입력 변환
            img_final = cv2.resize(imgWhite, (224, 224))
            img_array = np.expand_dims(np.array(img_final, dtype=np.float32) / 127.5 - 1, axis=0)
            
            # 예측
            pred = model.predict(img_array, verbose=0)
            idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            label = labels[idx]
            
            return {"prediction": [{"label": label, "confidence": confidence}]}
            
        else:
            return {"prediction": [{"label": "No Hand", "confidence": 0.0}]}
            
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# --- [핵심 수정] 정적 파일 서빙 설정 ---
# 404 에러를 해결하기 위해 경로를 명확히 분리했습니다.

static_path = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(static_path):
    # 1. HTML에서 src="/static/..."으로 요청하는 JS/CSS 파일들을 처리
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    # 2. 브라우저 주소창에 그냥 "/" (루트)로 접속했을 때 index.html 반환
    @app.get("/")
    async def read_index():
        return FileResponse(os.path.join(static_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Render 배포 환경 포트 설정
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)