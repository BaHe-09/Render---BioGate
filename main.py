from fastapi import FastAPI, HTTPException, Depends, status, Query, Header
from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime, timedelta
import logging
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# Configuración inicial
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de seguridad
SECRET_KEY = os.getenv("SECRET_KEY", "secret-key-for-dev-only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Inicializa la app FastAPI
app = FastAPI(title="Sistema de Control de Accesos API",
             description="API completa para gestión de usuarios, autenticación y control de accesos",
             version="1.0.0")

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    disabled: bool = False
    is_admin: bool = False

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
    id_rol: int = 2  # Por defecto rol de Usuario
    password: Optional[str] = None

class Dispositivo(BaseModel):
    id_dispositivo: int
    nombre: str
    tipo: str
    ubicacion: str
    estado: str

# --- Funciones de ayuda ---
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Header(...), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.execute(
        text("""
            SELECT c.id_cuenta as id, c.nombre_usuario as username, 
                   p.correo_electronico as email, 
                   CONCAT(p.nombre, ' ', p.apellido_paterno) as full_name,
                   r.nombre as role
            FROM cuentas c
            JOIN personas p ON c.id_persona = p.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE c.nombre_usuario = :username
        """),
        {"username": token_data.username}
    ).fetchone()
    
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.role != "Administrador":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permisos de administrador"
        )
    return current_user

# --- Endpoints de Autenticación ---
@app.post("/login/", response_model=Token)
async def login_for_access_token(user: UserLogin, db: Session = Depends(get_db)):
    # 1. Buscar usuario en la base de datos
    result = db.execute(
        text("""
            SELECT c.id_cuenta, c.contrasena_hash, c.nombre_usuario,
                   p.correo_electronico, p.nombre, p.apellido_paterno,
                   r.nombre as rol
            FROM cuentas c
            JOIN personas p ON c.id_persona = p.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE c.nombre_usuario = :username
        """),
        {"username": user.username}
    )
    user_db = result.fetchone()

    if not user_db or not verify_password(user.password, user_db.contrasena_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 2. Registrar el acceso en el historial
    try:
        db.execute(
            text("""
                INSERT INTO historial_accesos (id_persona, resultado, fecha)
                VALUES (
                    (SELECT id_persona FROM cuentas WHERE nombre_usuario = :username),
                    'Éxito',
                    NOW()
                )
            """),
            {"username": user.username}
        )
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error registrando acceso: {str(e)}")

    # 3. Generar token de acceso
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_db.nombre_usuario},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/registrar/", status_code=status.HTTP_201_CREATED)
def registrar_usuario(usuario: UsuarioRegistro, db: Session = Depends(get_db)):
    try:
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
        hashed_password = get_password_hash(usuario.cuenta.password)

        # Insertar persona
        result_persona = db.execute(
            text("""
                INSERT INTO personas (
                    nombre, apellido_paterno, apellido_materno, 
                    telefono, correo_electronico, fecha_registro, activo
                ) 
                VALUES (
                    :nombre, :apellido_paterno, :apellido_materno, 
                    :telefono, :correo, NOW(), TRUE
                )
                RETURNING id_persona
            """),
            {
                "nombre": usuario.persona.name,
                "apellido_paterno": usuario.persona.lastName,
                "apellido_materno": usuario.persona.secondLastName,
                "telefono": usuario.persona.phone,
                "correo": usuario.persona.email
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
                    '', 
                    NOW()
                )
            """),
            {
                "id_persona": id_persona,
                "nombre_usuario": nombre_usuario,
                "contrasena_hash": hashed_password
            }
        )

        db.commit()
        return {
            "status": "success",
            "id_persona": id_persona,
            "nombre_usuario": nombre_usuario
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

# --- Endpoints de Usuarios ---
@app.get("/usuarios/", response_model=List[Usuario])
def obtener_usuarios(
    search: Optional[str] = Query(None),
    incluir_admin: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
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
            WHERE (CONCAT(p.nombre, ' ', p.apellido_paterno) LIKE :search
                 OR p.correo_electronico LIKE :search
            """ + ("AND r.nombre != 'Administrador'" if not incluir_admin else "") + """
            ORDER BY p.nombre
        """)
        result = db.execute(query, {"search": f"%{search}%" if search else "%"})
        usuarios = result.fetchall()
        
        return [{
            "id_persona": u.id_persona,
            "nombre_completo": u.nombre_completo,
            "correo_electronico": u.correo_electronico,
            "telefono": u.telefono,
            "activo": u.activo,
            "id_rol": u.id_rol,
            "nombre_rol": u.nombre_rol
        } for u in usuarios]
        
    except Exception as e:
        logger.error(f"Error al obtener usuarios: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener la lista de usuarios"
        )

@app.post("/usuarios/", response_model=Usuario)
def crear_usuario(
    usuario: NuevoUsuario,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_admin_user)
):
    try:
        # Verificar si el correo ya existe
        correo_existente = db.execute(
            text("SELECT 1 FROM personas WHERE correo_electronico = :correo"),
            {"correo": usuario.correo_electronico}
        ).scalar()
        
        if correo_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El correo electrónico ya está registrado"
            )

        # Insertar persona
        result_persona = db.execute(
            text("""
                INSERT INTO personas (
                    nombre, apellido_paterno, apellido_materno, 
                    telefono, correo_electronico, fecha_registro, activo
                ) 
                VALUES (
                    :nombre, :apellido_paterno, :apellido_materno, 
                    :telefono, :correo, NOW(), TRUE
                )
                RETURNING id_persona
            """),
            {
                "nombre": usuario.nombre,
                "apellido_paterno": usuario.apellido_paterno,
                "apellido_materno": usuario.apellido_materno,
                "telefono": usuario.telefono,
                "correo": usuario.correo_electronico
            }
        )
        id_persona = result_persona.scalar_one()

        # Crear nombre de usuario a partir del correo
        nombre_usuario = usuario.correo_electronico.split('@')[0]
        password = usuario.password if usuario.password else nombre_usuario
        hashed_password = get_password_hash(password)

        # Insertar cuenta
        db.execute(
            text("""
                INSERT INTO cuentas (
                    id_persona, id_rol, nombre_usuario, 
                    contrasena_hash, sal, ultimo_acceso
                ) 
                VALUES (
                    :id_persona, 
                    :id_rol, 
                    :nombre_usuario, 
                    :contrasena_hash, 
                    '', 
                    NOW()
                )
            """),
            {
                "id_persona": id_persona,
                "id_rol": usuario.id_rol,
                "nombre_usuario": nombre_usuario,
                "contrasena_hash": hashed_password
            }
        )

        db.commit()
        
        # Devolver el usuario creado
        result = db.execute(
            text("""
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
            """),
            {"id": id_persona}
        )
        usuario_creado = result.fetchone()
        
        return usuario_creado

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear usuario: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al crear el usuario"
        )

@app.patch("/usuarios/{id_persona}", response_model=Usuario)
def actualizar_estado_usuario(
    id_persona: int, 
    usuario_update: UsuarioUpdate,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_admin_user)
):
    try:
        # Verificar si el usuario existe
        usuario = db.execute(
            text("""
                SELECT p.id_persona 
                FROM personas p
                WHERE p.id_persona = :id
            """),
            {"id": id_persona}
        ).fetchone()
        
        if not usuario:
            raise HTTPException(
                status_code=404,
                detail="Usuario no encontrado"
            )

        # Actualizar estado
        db.execute(
            text("UPDATE personas SET activo = :activo WHERE id_persona = :id"),
            {"activo": usuario_update.activo, "id": id_persona}
        )
        db.commit()
        
        # Devolver el usuario actualizado
        result = db.execute(
            text("""
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
            """),
            {"id": id_persona}
        )
        usuario_actualizado = result.fetchone()
        
        return usuario_actualizado
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar usuario: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al actualizar el estado del usuario"
        )

# --- Endpoints de Historial ---
@app.get("/historial-accesos/", response_model=List[HistorialAcceso])
def obtener_historial_accesos(
    filtros: HistorialFiltrado = Depends(),
    limite: int = Query(20, gt=0, le=100),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
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
def obtener_detalle_acceso(
    id_acceso: int, 
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
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

# --- Endpoints de Dispositivos ---
@app.get("/dispositivos/", response_model=List[Dispositivo])
def obtener_dispositivos(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        result = db.execute(
            text("""
                SELECT id_dispositivo, nombre, tipo, ubicacion, estado
                FROM dispositivos
                ORDER BY nombre
            """)
        )
        dispositivos = result.fetchall()
        
        return [{
            "id_dispositivo": d.id_dispositivo,
            "nombre": d.nombre,
            "tipo": d.tipo,
            "ubicacion": d.ubicacion,
            "estado": d.estado
        } for d in dispositivos]
        
    except Exception as e:
        logger.error(f"Error al obtener dispositivos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener la lista de dispositivos"
        )

# --- Endpoints de Utilidades ---
@app.get("/generate-password/")
def generate_password(password: str):
    """Genera un hash bcrypt para contraseñas (uso en desarrollo)"""
    hashed = get_password_hash(password)
    return {
        "original": password,
        "hashed": hashed,
        "warning": "No usar en producción"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api", "version": "1.0.0"}

@app.get("/me", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user
