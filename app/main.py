from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import cv2
import numpy as np
import tempfile
import os
from keras_facenet import FaceNet
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

yolo = YOLO('models/yolov8n-face-lindevs.pt')
facenet = FaceNet()
conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))

def extraer_rostro(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")
    results = yolo(img)
    if not results or len(results[0].boxes) == 0:
        raise HTTPException(status_code=404, detail="No se detectaron rostros")
    boxes = results[0].boxes
    main_box = boxes[np.argmax(boxes.conf.cpu().numpy())]
    x1, y1, x2, y2 = map(int, main_box.xyxy[0].cpu().numpy())
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        raise HTTPException(status_code=400, detail="Área de rostro inválida")
    return face

def generar_embedding(face: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
    return embedding

def buscar_coincidencia(embedding: np.ndarray, threshold: float=0.7):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT p.id_persona, p.nombre, p.apellido_paterno, p.apellido_materno, p.correo_electronico,
                   1 - (v.vector <=> %s::vector) as similitud
            FROM vectores_identificacion v
            JOIN personas p ON v.id_persona = p.id_persona
            WHERE 1 - (v.vector <=> %s::vector) > %s
            ORDER BY similitud DESC
            LIMIT 1
        """, (embedding.tolist(), embedding.tolist(), threshold))
        return cursor.fetchone()

@app.post("/clasificar")
async def clasificar(file: UploadFile = File(...), threshold: float = Query(0.7, ge=0.5, le=0.9)):
    try:
        # Guardar imagen temporalmente con tempfile
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        face = extraer_rostro(temp_path)
        embedding = generar_embedding(face)
        resultado = buscar_coincidencia(embedding, threshold)
        
        os.remove(temp_path)  # limpiar archivo temporal
        
        if resultado:
            id_persona, nombre, apellido_p, apellido_m, correo, similitud = resultado
            return {
                "status": "success",
                "persona": {
                    "id": id_persona,
                    "nombre": nombre,
                    "apellido_paterno": apellido_p,
                    "apellido_materno": apellido_m,
                    "correo": correo,
                },
                "similitud": similitud
            }
        else:
            return {"status": "no_match"}
    except Exception as e:
        print("Error interno:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
