from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API de Reconocimiento Facial funcionando"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # === Carga diferida de modelos ===
        from ultralytics import YOLO
        from keras.models import load_model

        yolomodel = YOLO("app/models/yolov8n-face.pt")
        facenet = load_model("app/models/facenet_keras.h5")

        # === Detección de rostros con YOLO ===
        results = yolomodel.predict(source=np.array(image), conf=0.5, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return JSONResponse(status_code=404, content={"message": "No se detectó ningún rostro."})

        # Obtener la primer detección
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        face_crop = image.crop((x1, y1, x2, y2)).resize((160, 160))

        # === Vector de embedding con FaceNet ===
        face_array = np.asarray(face_crop).astype("float32")
        mean, std = face_array.mean(), face_array.std()
        face_array = (face_array - mean) / std
        face_array = np.expand_dims(face_array, axis=0)
        embedding = facenet.predict(face_array)[0]

        return {
            "message": "Embeddings generados correctamente",
            "embedding": embedding.tolist()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
