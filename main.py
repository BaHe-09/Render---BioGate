from fastapi import FastAPI, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
import logging
from fastapi.middleware.cors import CORSMiddleware
from database import get_db

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa la app FastAPI
app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe esto a tus dominios
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class UserLogin(BaseModel):
    username: str
    password: str

class RegistroPersona(BaseModel):
    name: str  # Nombre(s)
    lastName: str  # Primer apellido
    secondLastName: Optional[str] = None  # Segundo apellido (opcional)
    phone: str  # Teléfono completo (código + número)
    email: EmailStr  # Correo electrónico
    genero: Optional[str] = None

class RegistroCuenta(BaseModel):
    password: str  # Contraseña
    confirmPassword: str  # Confirmación de contraseña

    @validator('confirmPassword')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Las contraseñas no coinciden')
        return v

class UsuarioRegistro(BaseModel):
    persona: RegistroPersona
    cuenta: RegistroCuenta

class UsuarioResponse(BaseModel):
    id_persona: int
    nombre_completo: str
    correo_electronico: Optional[str] = None
    telefono: str
    genero: Optional[str] = None
    activo: bool
    rol: str

class ActualizarEstadoRequest(BaseModel):
    activo: bool

# --- Endpoints ---
@app.get("/")
def read_root():
    return {
        "message": "API de autenticación funcionando",
        "status": "active",
        "endpoints": {
            "login": "POST /login/",
            "register": "POST /registrar/",
            "usuarios": "GET /personas/",
            "usuario": "GET /personas/{id}",
            "actualizar_estado": "PUT /personas/{id}/estado",
            "docs": "/docs"
        }
    }

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
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )

@app.post("/registrar/", status_code=status.HTTP_201_CREATED)
def registrar_usuario(usuario: UsuarioRegistro, db: Session = Depends(get_db)):
    try:
        logger.info(f"Intento de registro para: {usuario.persona.email}")
        
        # Verificar si el correo ya existe
        correo_existente = db.execute(
            text("SELECT 1 FROM personas WHERE correo_electronico = :correo"),
            {"correo": usuario.persona.email}
        ).scalar()
        
        if correo_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El correo electrónico ya está registrado"
            )

        # Verificar si el nombre de usuario ya existe
        nombre_usuario = usuario.persona.email.split('@')[0]
        usuario_existente = db.execute(
            text("SELECT 1 FROM cuentas WHERE nombre_usuario = :username"),
            {"username": nombre_usuario}
        ).scalar()
        
        if usuario_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre de usuario ya está en uso"
            )

        # Hashear contraseña
        hashed_password = bcrypt.hashpw(
            usuario.cuenta.password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        # Insertar persona
        result_persona = db.execute(
            text("""
                INSERT INTO personas (
                    nombre, apellido_paterno, apellido_materno, 
                    telefono, correo_electronico, genero, fecha_registro, activo
                ) 
                VALUES (
                    :nombre, :apellido_paterno, :apellido_materno, 
                    :telefono, :correo, :genero, :fecha_registro, TRUE
                )
                RETURNING id_persona
            """),
            {
                "nombre": usuario.persona.name,
                "apellido_paterno": usuario.persona.lastName,
                "apellido_materno": usuario.persona.secondLastName,
                "telefono": usuario.persona.phone,
                "correo": usuario.persona.email,
                "genero": usuario.persona.genero,
                "fecha_registro": datetime.now()
            }
        )
        id_persona = result_persona.scalar_one()

        # Insertar cuenta
        db.execute(
            text("""
                INSERT INTO cuentas (
                    id_persona, id_rol, nombre_usuario, 
                    contrasena_hash, sal, ultimo_acceso
                ) 
                VALUES (
                    :id_persona, 
                    1,  -- Rol de Administrador
                    :nombre_usuario, 
                    :contrasena_hash, 
                    '',  -- Sal (ya incluida en bcrypt)
                    :ultimo_acceso
                )
            """),
            {
                "id_persona": id_persona,
                "nombre_usuario": nombre_usuario,
                "contrasena_hash": hashed_password,
                "ultimo_acceso": datetime.now()
            }
        )

        db.commit()
        logger.info(f"Usuario administrador registrado exitosamente: {usuario.persona.email}")

        return {
            "status": "success",
            "id_persona": id_persona,
            "nombre_usuario": nombre_usuario,
            "message": "Usuario administrador registrado exitosamente"
        }

    except HTTPException:
        db.rollback()
        raise
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )

@app.get("/personas/", response_model=List[UsuarioResponse])
def obtener_personas(
    rol: Optional[str] = Query(None),
    activo: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    try:
        query = text("""
            SELECT 
                p.id_persona,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                p.correo_electronico,
                p.telefono,
                p.genero,
                p.activo,
                r.nombre as rol
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE r.nombre != 'Administrador'
            AND (:rol IS NULL OR r.nombre = :rol)
            AND (:activo IS NULL OR p.activo = :activo)
            ORDER BY p.nombre
        """)
        result = db.execute(query, {"rol": rol, "activo": activo})
        return result.fetchall()
        
    except Exception as e:
        logger.error(f"Error al obtener personas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener la lista de personas"
        )

@app.get("/personas/{id_persona}", response_model=UsuarioResponse)
def obtener_persona(id_persona: int, db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT 
                p.id_persona,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                p.correo_electronico,
                p.telefono,
                p.genero,
                p.activo,
                r.nombre as rol
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE p.id_persona = :id_persona
            AND r.nombre != 'Administrador'
        """)
        result = db.execute(query, {"id_persona": id_persona})
        persona = result.fetchone()
        
        if not persona:
            raise HTTPException(
                status_code=404,
                detail="Persona no encontrada o no tiene permisos"
            )
            
        return persona
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener persona: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener la información de la persona"
        )

@app.put("/personas/{id_persona}/estado")
def actualizar_estado(
    id_persona: int,
    datos: ActualizarEstadoRequest,
    db: Session = Depends(get_db)
):
    try:
        # Verificar que el usuario existe y no es administrador
        query = text("""
            SELECT 1 FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE p.id_persona = :id_persona
            AND r.nombre != 'Administrador'
        """)
        existe = db.execute(query, {"id_persona": id_persona}).scalar()
        
        if not existe:
            raise HTTPException(
                status_code=404,
                detail="Persona no encontrada o no tiene permisos"
            )
        
        # Actualizar estado
        db.execute(
            text("UPDATE personas SET activo = :activo WHERE id_persona = :id_persona"),
            {"activo": datos.activo, "id_persona": id_persona}
        )
        db.commit()
        
        return {
            "status": "success",
            "activo": datos.activo,
            "message": "Estado actualizado correctamente"
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar estado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al actualizar el estado de la persona"
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
