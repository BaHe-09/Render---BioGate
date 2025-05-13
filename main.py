from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import numpy as np
from models import process_image, get_face_embedding
from database import get_db
from typing import List, Dict, Any
import os
import uvicorn

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos pre-cargados (se inicializarán al iniciar la app)
yolo_model = None
facenet_model = None

@app.on_event("startup")
async def startup_event():
    global yolo_model, facenet_model
    # Inicializa los modelos aquí
    from models import load_models
    yolo_model, facenet_model = load_models()

@app.post("/recognize")
async def recognize_face(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    threshold: float = 0.7
):
    try:
        # 1. Procesar imagen con YOLO
        image_bytes = await file.read()
        face_image = process_image(image_bytes, yolo_model)
        
        if face_image is None:
            return {"status": "error", "message": "No face detected"}
        
        # 2. Obtener embedding con FaceNet
        embedding = get_face_embedding(face_image, facenet_model)
        
        # 3. Buscar en la base de datos
        closest_match = find_closest_match(db, embedding, threshold)
        
        if closest_match:
            return {
                "status": "success",
                "match": True,
                "confidence": float(1 - closest_match["distance"]),
                "person": closest_match["persona"],
                "vector_id": closest_match["id_vector"]
            }
        else:
            return {
                "status": "success",
                "match": False,
                "message": "Unknown face"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_closest_match(db: Session, embedding: np.ndarray, threshold: float) -> Dict[str, Any]:
    # Convertir el embedding a una lista para la consulta SQL
    embedding_list = embedding.tolist()
    
    # Consulta para encontrar el vector más cercano
    query = """
        SELECT 
            v.id_vector,
            v.vector <-> %s AS distance,
            p.*
        FROM 
            vectores_identificacion v
        JOIN 
            personas p ON v.id_persona = p.id_persona
        ORDER BY 
            vector <-> %s
        LIMIT 1
    """
    
    result = db.execute(query, (embedding_list, embedding_list)).fetchone()
    
    if result and result["distance"] <= threshold:
        return {
            "id_vector": result["id_vector"],
            "distance": result["distance"],
            "persona": {
                "id_persona": result["id_persona"],
                "nombre": result["nombre"],
                "apellido_paterno": result["apellido_paterno"],
                "apellido_materno": result["apellido_materno"],
                "telefono": result["telefono"],
                "correo_electronico": result["correo_electronico"]
            }
        }
    return None

@app.get("/")
def read_root():
    return {"message": "Face Recognition API"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Reduce workers para ahorrar memoria
        log_level="info"
    )
