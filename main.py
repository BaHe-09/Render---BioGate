import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from facenet_utils import load_facenet_model, get_embedding, compare_with_db
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Face Recognition API", 
              description="API para comparar rostros con embeddings en Neontech DB")

# Cargar modelo Facenet al iniciar
facenet_model = load_facenet_model()

# Configuración de Neontech DB
DB_CONFIG = {
    "dbname": os.getenv("NEON_DB_NAME"),
    "user": os.getenv("NEON_DB_USER"),
    "password": os.getenv("NEON_DB_PASSWORD"),
    "host": os.getenv("NEON_DB_HOST"),
    "port": os.getenv("NEON_DB_PORT")
}

@app.get("/")
def health_check():
    return {"status": "OK", "message": "Face Recognition API with Facenet"}

@app.post("/compare-face")
async def compare_face(
    file: UploadFile = File(...),
    threshold: float = 0.7,
    top_k: int = 5
):
    try:
        # Leer y procesar imagen
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Obtener embedding con Facenet
        embedding = get_embedding(facenet_model, img)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Comparar con la base de datos
        matches = compare_with_db(DB_CONFIG, embedding, threshold, top_k)
        
        return JSONResponse(content={
            "status": "success",
            "matches": matches,
            "embedding_size": len(embedding)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10001))  # Puerto diferente al de la otra API
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
