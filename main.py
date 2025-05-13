import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Fuerza CPU
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# Carga el modelo con seguridad
model = YOLO('yolov8n-face-lindevs.pt', task='detect')

@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Detección
        results = model.predict(img, imgsz=320)
        
        # Dibujar bounding boxes
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convertir a JPEG
        _, img_encoded = cv2.imencode('.jpg', img)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
    
    except Exception as e:
        return {"error": str(e)}
