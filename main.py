from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# Carga el modelo YOLO al iniciar
modelo_yolo = YOLO('yolov8n-face-lindevs.pt')

@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    try:
        # 1. Leer la imagen
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. Detectar rostros
        results = modelo_yolo(img)
        
        # 3. Dibujar bounding boxes
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 4. Convertir imagen a bytes para la respuesta
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
