import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desactiva GPU
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io
import torch  # Añade esto

# Configura PyTorch para carga segura
torch.serialization.add_safe_globals([torch.nn.Module])

app = FastAPI()

# Carga YOLO con modo seguro
modelo_yolo = YOLO('yolov8n-face-lindevs.pt', task='detect').to('cpu')

@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detección (usa .predict() en lugar de llamar al modelo directamente)
        results = modelo_yolo.predict(img, imgsz=320, conf=0.5)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        _, img_encoded = cv2.imencode('.jpg', img)
        return StreamingResponse(
            io.BytesIO(img_encoded.tobytes()),
            media_type="image/jpeg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/")
def home():
    return {"message": "API de detección de rostros con YOLO"}
