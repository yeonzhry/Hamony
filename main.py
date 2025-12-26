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
# import mysql.connector
# from mysql.connector import errorcode
import datetime
import json
import logging
import os

DB_CONFIG = {
    'user': 'admin', 
    'password': 'Zmt7rwtt64',
    'host': 'database-1.cv42kwy26xf8.ap-northeast-2.rds.amazonaws.com', 
    'database': 'fastapi-ca'      
}

CURRENT_UserId="kyung"

#인덱스 저장 함수 
def save_prediction_to_db_mysql(user_id, prediction_index): 
    conn = None 
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        query = ("INSERT INTO predictions "
                 "(user_id, prediction_index, timestamp) "
                 "VALUES (%s, %s, %s)")
        
        data = (user_id, int(prediction_index), timestamp)

        cursor.execute(query, data)
        conn.commit()
        print(f"user: '{user_id}' index prediction: {prediction_index}") 

    except mysql.connector.Error as err:
        print(f"DB 오류")
    except Exception as e:
        print(f"❌ 예측 저장 중 예상치 못한 오류 발생: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 (프론트엔드와 구분하기 위해 /api prefix 사용)
api_app = FastAPI()

# 모델 로드
model = tf.keras.models.load_model("keras_model.h5")
labels = ["Re", "Mi", "Fa", "Sol", "La", "Ti", "Do"]

# 손 감지기 초기화
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

@api_app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # 이미지 열기
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # PIL을 OpenCV 형식으로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 손 감지
        hands, _ = detector.findHands(img_cv)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # 손 영역 크롭 (원본 코드와 동일한 전처리)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img_cv[y-offset:y+h+offset, x-offset:x+w+offset]
            
            if imgCrop.size > 0:  # 크롭된 이미지가 유효한지 확인
                imgCropShape = imgCrop.shape
                aspectRatio = h/w
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal+wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                
                # 모델 입력 형식으로 변환 (224x224)
                img_final = cv2.resize(imgWhite, (224, 224))
                img_array = np.expand_dims(np.array(img_final)/255.0, axis=0)
                
                # 예측
                pred = model.predict(img_array, verbose=0)
                idx = int(np.argmax(pred))
                confidence = float(np.max(pred))
                label = labels[idx]
                
                print(f"Hand detected! Predicted: {label}, Confidence: {confidence:.3f}")
                
                return {"prediction": [{"label": label, "confidence": confidence}]}
            else:
                print("Invalid crop - no hand detected properly")
                return {"prediction": [{"label": "No Hand", "confidence": 0.0}]}
        else:
            print("No hand detected in image")
            return {"prediction": [{"label": "No Hand", "confidence": 0.0}]}
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"error": str(e)}

# 프론트엔드 빌드 파일 서빙 설정
frontend_build_path = os.path.join(os.path.dirname(__file__), "../Front/build")

# 루트 경로 핸들러 먼저 정의 (FastAPI는 정의 순서대로 매칭하므로 중요)
if os.path.exists(frontend_build_path):
    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(frontend_build_path, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Frontend build not found. Please build the frontend first."}

# API 라우터를 메인 앱에 마운트 (마운트는 높은 우선순위)
app.mount("/api", api_app)

# 프론트엔드 빌드 파일 서빙
if os.path.exists(frontend_build_path):
    # 정적 파일 서빙 (CSS, JS 등) - 마운트는 높은 우선순위
    static_path = os.path.join(frontend_build_path, "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # 루트 경로의 모든 정적 파일들 (images, favicon.ico, manifest.json 등)
    # React Router를 지원하기 위해 catch-all 라우트 사용
    # 주의: 마운트된 경로(/api, /static)는 이 라우트보다 우선 처리됨
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # 파일 경로 생성
        file_path = os.path.join(frontend_build_path, full_path)
        
        # 파일이 존재하고 실제 파일인 경우
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # 디렉토리이거나 파일이 없는 경우 index.html 반환 (React Router)
        return FileResponse(os.path.join(frontend_build_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
