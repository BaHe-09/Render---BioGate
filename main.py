from fastapi import FastAPI, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt
from pydantic import BaseModel, EmailStr, Field, validator
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
    name: str
    lastName: str
    secondLastName: Optional[str] = None
    phone: str
    email: EmailStr

class RegistroCuenta(BaseModel):
    password: str
    confirmPassword: str

    @validator('confirmPassword')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Las contraseñas no coinciden')
        return v

class UsuarioRegistro(BaseModel):
    persona: RegistroPersona
    cuenta: RegistroCuenta

class Usuario(BaseModel):
    id_persona: int
    nombre_completo: str
    correo_electronico: Optional[str] = None
    telefono: Optional[str] = None
    activo: bool
    id_rol: int
    nombre_rol: str
    es_admin: bool = Field(False, description="Indica si el usuario es administrador")

    @validator('es_admin', pre=True, always=True)
    def set_es_admin(cls, v, values):
        return values.get('nombre_rol', '').lower() == 'administrador'

class UsuarioUpdate(BaseModel):
    activo: bool

class NuevoUsuario(BaseModel):
    nombre: str
    apellido_paterno: str
    apellido_materno: Optional[str] = None
    telefono: str
    correo_electronico: EmailStr
    id_rol: int = Field(2, description="ID del rol (2 para Usuario por defecto)")
    password: Optional[str] = None

class Rol(BaseModel):
    id_rol: int
    nombre: str
    descripcion: Optional[str] = None

# --- Endpoints ---

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "API de gestión de usuarios"}

# --- Autenticación ---
@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        logger.info(f"Intento de login para: {user.username}")
        
        query = text("""
            SELECT c.id_cuenta, c.contrasena_hash, r.nombre as rol
            FROM cuentas c
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE c.nombre_usuario = :username
            LIMIT 1
        """)
        result = db.execute(query, {"username": user.username})
        user_db = result.fetchone()

        if not user_db:
            logger.warning("Usuario no encontrado")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        if not bcrypt.checkpw(user.password.encode('utf-8'), user_db.contrasena_hash.encode('utf-8')):
            logger.warning("Contraseña incorrecta")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        logger.info("Autenticación exitosa")
        return {
            "status": "success",
            "user_id": user_db.id_cuenta,
            "rol": user_db.rol,
            "message": "Autenticación exitosa"
        }

    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# --- Gestión de Usuarios ---
@app.get("/usuarios/", response_model=List[Usuario])
def obtener_usuarios(
    search: Optional[str] = Query(None),
    incluir_admin: bool = Query(False),
    db: Session = Depends(get_db)
):
    try:
        query_params = {"search": f"%{search}%" if search else "%"}
        
        query = text(f"""
            SELECT 
                p.id_persona,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                p.correo_electronico,
                p.telefono,
                p.activo,
                c.id_rol,
                r.nombre as nombre_rol
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE (CONCAT(p.nombre, ' ', p.apellido_paterno) LIKE :search
                OR p.correo_electronico LIKE :search)
            {'AND r.nombre != "Administrador"' if not incluir_admin else ''}
            ORDER BY p.nombre
        """)
        
        result = db.execute(query, query_params)
        return [
            {
                **dict(u),
                "es_admin": u.nombre_rol == "Administrador"
            } 
            for u in result.fetchall()
        ]
        
    except Exception as e:
        logger.error(f"Error al obtener usuarios: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener usuarios")

@app.get("/usuarios/{id_persona}", response_model=Usuario)
def obtener_usuario(id_persona: int, db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT 
                p.id_persona,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                p.correo_electronico,
                p.telefono,
                p.activo,
                c.id_rol,
                r.nombre as nombre_rol
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE p.id_persona = :id
        """)
        
        usuario = db.execute(query, {"id": id_persona}).fetchone()
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
            
        return {**dict(usuario), "es_admin": usuario.nombre_rol == "Administrador"}
        
    except Exception as e:
        logger.error(f"Error al obtener usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener usuario")

@app.patch("/usuarios/{id_persona}", response_model=Usuario)
def actualizar_usuario(
    id_persona: int, 
    update: UsuarioUpdate,
    db: Session = Depends(get_db)
):
    try:
        # Verificar si el usuario existe
        usuario = db.execute(
            text("SELECT 1 FROM personas WHERE id_persona = :id"),
            {"id": id_persona}
        ).fetchone()
        
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Actualizar estado
        db.execute(
            text("UPDATE personas SET activo = :activo WHERE id_persona = :id"),
            {"activo": update.activo, "id": id_persona}
        )
        db.commit()
        
        # Devolver usuario actualizado
        return obtener_usuario(id_persona, db)
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al actualizar usuario")

@app.post("/usuarios/", status_code=status.HTTP_201_CREATED, response_model=Usuario)
def crear_usuario(usuario: NuevoUsuario, db: Session = Depends(get_db)):
    try:
        # Verificar correo único
        if db.execute(
            text("SELECT 1 FROM personas WHERE correo_electronico = :email"),
            {"email": usuario.correo_electronico}
        ).fetchone():
            raise HTTPException(status_code=400, detail="El correo ya está registrado")

        # Insertar persona
        result = db.execute(
            text("""
                INSERT INTO personas (
                    nombre, apellido_paterno, apellido_materno,
                    telefono, correo_electronico, fecha_registro, activo
                ) VALUES (
                    :nombre, :apellido_paterno, :apellido_materno,
                    :telefono, :email, NOW(), TRUE
                ) RETURNING id_persona
            """),
            {
                "nombre": usuario.nombre,
                "apellido_paterno": usuario.apellido_paterno,
                "apellido_materno": usuario.apellido_materno,
                "telefono": usuario.telefono,
                "email": usuario.correo_electronico
            }
        )
        id_persona = result.fetchone().id_persona

        # Crear cuenta
        username = usuario.correo_electronico.split('@')[0]
        password = usuario.password if usuario.password else username
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        db.execute(
            text("""
                INSERT INTO cuentas (
                    id_persona, id_rol, nombre_usuario,
                    contrasena_hash, sal, ultimo_acceso
                ) VALUES (
                    :id_persona, :id_rol, :username,
                    :hashed_pw, '', NOW()
                )
            """),
            {
                "id_persona": id_persona,
                "id_rol": usuario.id_rol,
                "username": username,
                "hashed_pw": hashed_pw
            }
        )
        db.commit()
        
        return obtener_usuario(id_persona, db)
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al crear usuario")

# --- Gestión de Roles ---
@app.get("/roles/", response_model=List[Rol])
def obtener_roles(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT id_rol, nombre, descripcion FROM roles"))
        return [dict(r) for r in result.fetchall()]
    except Exception as e:
        logger.error(f"Error al obtener roles: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener roles")

# --- Utilitarios ---
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}

@app.get("/generate-password/")
def generate_password(password: str = Query(...)):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return {"hashed": hashed.decode('utf-8')}
