from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import os

# --- 앱 설정 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 모델 로드 (호환성 해결) ---
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

try:
    model = tf.keras.models.load_model("keras_model.h5", 
                                       custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
                                       compile=False)
except:
    model = tf.keras.models.load_model("keras_model.h5", compile=False)

labels = ["Re", "Mi", "Fa", "Sol", "La", "Ti", "Do"]
detector = HandDetector(maxHands=1)

# --- API ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        hands, _ = detector.findHands(img_cv)
        if hands:
            # (간소화된 예측 로직)
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img_cv[y-20:y+h+20, x-20:x+w+20]
            
            if imgCrop.size == 0: return {"prediction": [{"label": "No Hand"}]}
            
            imgResize = cv2.resize(imgCrop, (224, 224))
            img_array = np.expand_dims(np.array(imgResize, dtype=np.float32) / 127.5 - 1, axis=0)
            
            pred = model.predict(img_array, verbose=0)
            idx = int(np.argmax(pred))
            return {"prediction": [{"label": labels[idx], "confidence": float(np.max(pred))}]}
        return {"prediction": [{"label": "No Hand"}]}
    except Exception as e:
        return {"error": str(e)}

# --- 여기가 핵심! 정적 파일 연결 ---
# static 폴더를 루트('/')에 연결합니다.
static_path = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
else:
    # static 폴더가 없을 때만 이 메시지가 뜸 (근데 님은 폴더가 있으니 이게 뜨면 안 됨)
    @app.get("/")
    def error():
        return {"error": "static 폴더를 찾을 수 없습니다. 폴더 구조를 확인하세요."}