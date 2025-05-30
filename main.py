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

class HistorialAcceso(BaseModel):
    id_acceso: int
    nombre_completo: str
    fecha: str
    resultado: str
    dispositivo: str
    foto_url: Optional[str] = None

class HistorialFiltrado(BaseModel):
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    resultado: Optional[str] = None
    nombre: Optional[str] = None

class PersonaResponse(BaseModel):
    id_persona: int
    nombre: str
    apellido_paterno: str
    apellido_materno: Optional[str] = None
    correo_electronico: Optional[str] = None
    telefono: Optional[str] = None
    activo: bool
    fecha_registro: datetime

    class Config:
        from_attributes = True  # Esto permite la conversión desde ORM models

class ActualizarEstadoPersona(BaseModel):
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
            "historial": "GET /historial-accesos/",
            "generate_password": "GET /generate-password/",
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
                    telefono, correo_electronico, fecha_registro, activo
                ) 
                VALUES (
                    :nombre, :apellido_paterno, :apellido_materno, 
                    :telefono, :correo, :fecha_registro, TRUE
                )
                RETURNING id_persona
            """),
            {
                "nombre": usuario.persona.name,
                "apellido_paterno": usuario.persona.lastName,
                "apellido_materno": usuario.persona.secondLastName,
                "telefono": usuario.persona.phone,
                "correo": usuario.persona.email,
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

@app.get("/historial-accesos/", response_model=List[HistorialAcceso])
def obtener_historial_accesos(
    filtros: HistorialFiltrado = Depends(),
    limite: int = Query(20, gt=0, le=100),
    db: Session = Depends(get_db)
):
    try:
        query_params = {
            "limite": limite,
            "nombre": f"%{filtros.nombre}%" if filtros.nombre else "%"
        }

        # Construir la consulta base
        query = text("""
            SELECT 
                ha.id_acceso,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                TO_CHAR(ha.fecha, 'DD/MM/YYYY – HH:MI AM') as fecha,
                CASE 
                    WHEN ha.resultado = 'Éxito' THEN 'PERMITIDO'
                    ELSE 'DENEGADO'
                END as resultado,
                COALESCE(d.nombre, 'Desconocido') as dispositivo,
                ha.foto_url
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            WHERE CONCAT(p.nombre, ' ', p.apellido_paterno) LIKE :nombre
        """)

        # Añadir filtros de fecha si están presentes
        if filtros.fecha_inicio and filtros.fecha_fin:
            query = text(str(query) + " AND ha.fecha BETWEEN :fecha_inicio AND :fecha_fin")
            query_params.update({
                "fecha_inicio": filtros.fecha_inicio,
                "fecha_fin": filtros.fecha_fin
            })
        
        # Añadir filtro de resultado si está presente
        if filtros.resultado:
            if filtros.resultado.upper() == 'PERMITIDO':
                query = text(str(query) + " AND ha.resultado = 'Éxito'")
            elif filtros.resultado.upper() == 'DENEGADO':
                query = text(str(query) + " AND ha.resultado != 'Éxito'")

        # Ordenar y limitar
        query = text(str(query) + " ORDER BY ha.fecha DESC LIMIT :limite")

        result = db.execute(query, query_params)
        historial = result.fetchall()
        
        return [{
            "id_acceso": item.id_acceso,
            "nombre_completo": item.nombre_completo,
            "fecha": item.fecha,
            "resultado": item.resultado,
            "dispositivo": item.dispositivo,
            "foto_url": item.foto_url
        } for item in historial]
        
    except Exception as e:
        logger.error(f"Error al obtener historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener el historial de accesos"
        )

@app.get("/historial-accesos/{id_acceso}", response_model=HistorialAcceso)
def obtener_detalle_acceso(id_acceso: int, db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT 
                ha.id_acceso,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                TO_CHAR(ha.fecha, 'DD/MM/YYYY – HH:MI AM') as fecha,
                CASE 
                    WHEN ha.resultado = 'Éxito' THEN 'PERMITIDO'
                    ELSE 'DENEGADO'
                END as resultado,
                COALESCE(d.nombre, 'Desconocido') as dispositivo,
                ha.foto_url,
                ha.confianza,
                ha.metadatos
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            WHERE ha.id_acceso = :id_acceso
        """)
        result = db.execute(query, {"id_acceso": id_acceso})
        acceso = result.fetchone()

        if not acceso:
            raise HTTPException(
                status_code=404,
                detail="Registro de acceso no encontrado"
            )

        return {
            "id_acceso": acceso.id_acceso,
            "nombre_completo": acceso.nombre_completo,
            "fecha": acceso.fecha,
            "resultado": acceso.resultado,
            "dispositivo": acceso.dispositivo,
            "foto_url": acceso.foto_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener detalle de acceso: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener el detalle del acceso"
        )

@app.get("/generate-password/")
def generate_password(password: str):
    """Genera un hash bcrypt para contraseñas (uso en desarrollo)"""
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return {
        "original": password,
        "hashed": hashed.decode('utf-8'),
        "warning": "No usar en producción"
    }

@app.get("/personas/", response_model=List[PersonaResponse])
def obtener_personas(db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT 
                id_persona,
                nombre,
                apellido_paterno,
                apellido_materno,
                correo_electronico,
                telefono,
                activo,
                fecha_registro
            FROM personas
            ORDER BY nombre, apellido_paterno
        """)
        result = db.execute(query)
        personas = result.fetchall()
        
        return [{
            "id_persona": p.id_persona,
            "nombre": p.nombre,
            "apellido_paterno": p.apellido_paterno,
            "apellido_materno": p.apellido_materno,
            "correo_electronico": p.correo_electronico,
            "telefono": p.telefono,
            "activo": p.activo,
            "fecha_registro": p.fecha_registro
        } for p in personas]
        
    except Exception as e:
        logger.error(f"Error al obtener personas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener la lista de personas"
        )

@app.put("/personas/{id_persona}/estado", status_code=status.HTTP_200_OK)
def actualizar_estado_persona(
    id_persona: int,
    estado: ActualizarEstadoPersona,
    db: Session = Depends(get_db)
):
    try:
        # Verificar si la persona existe
        persona_existente = db.execute(
            text("SELECT 1 FROM personas WHERE id_persona = :id"),
            {"id": id_persona}
        ).scalar()
        
        if not persona_existente:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persona no encontrada"
            )

        # Actualizar estado
        db.execute(
            text("""
                UPDATE personas 
                SET activo = :activo 
                WHERE id_persona = :id_persona
            """),
            {
                "id_persona": id_persona,
                "activo": estado.activo
            }
        )
        db.commit()
        
        return {
            "status": "success",
            "message": "Estado actualizado correctamente"
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar estado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al actualizar el estado"
        )
        
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
