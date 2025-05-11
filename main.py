from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt
from pydantic import BaseModel
from database import get_db
import logging
from fastapi.middleware.cors import CORSMiddleware
import os

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa la app FastAPI
app = FastAPI()

# Configura CORS (¡Ajusta los orígenes en producción!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (solo para desarrollo)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para los datos de login
class UserLogin(BaseModel):
    username: str
    password: str

# Endpoint raíz
@app.get("/")
def read_root():
    return {
        "message": "API de autenticación funcionando",
        "status": "active",
        "endpoints": {
            "login": "POST /login/",
            "generate_password": "GET /generate-password/",
            "docs": "/docs"
        }
    }

# Endpoint de autenticación
@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        logger.info(f"Intento de login para: {user.username}")
        
        # 1. Buscar usuario en la base de datos
        query = text("""
            SELECT id_cuenta, contrasena_hash 
            FROM cuentas 
            WHERE nombre_usuario = :username
            LIMIT 1
        """)
        result = db.execute(query, {"username": user.username})
        user_db = result.fetchone()

        if not user_db:
            logger.warning("Usuario no encontrado")
            raise HTTPException(
                status_code=401,
                detail="Credenciales inválidas",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 2. Verificar contraseña con bcrypt
        if not bcrypt.checkpw(
            user.password.encode('utf-8'),
            user_db.contrasena_hash.encode('utf-8')
        ):
            logger.warning("Contraseña incorrecta")
            raise HTTPException(
                status_code=401,
                detail="Credenciales inválidas",
                headers={"WWW-Authenticate": "Bearer"}
            )

        logger.info("Autenticación exitosa")
        return {
            "status": "success",
            "user_id": user_db.id_cuenta,
            "message": "Autenticación exitosa"
        }

    except HTTPException:
        raise  # Re-lanza excepciones HTTP manejadas
        
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )

# Endpoint para generar hashes de contraseña (solo desarrollo)
@app.get("/generate-password/")
def generate_password(password: str):
    """Genera un hash bcrypt para contraseñas (uso en desarrollo)"""
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return {
        "original": password,
        "hashed": hashed.decode('utf-8'),
        "warning": "No usar en producción"
    }

# Health check para Render (evita hibernación)
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
