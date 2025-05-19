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

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODELOS PYDANTIC ==========
class UserLogin(BaseModel):
    username: str
    password: str

class PersonaBase(BaseModel):
    nombre: str
    apellido_paterno: str
    apellido_materno: Optional[str] = None
    telefono: str
    correo_electronico: EmailStr
    genero: Optional[str] = None

class CuentaBase(BaseModel):
    password: str
    confirmPassword: str

    @validator('confirmPassword')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Las contraseñas no coinciden')
        return v

class UsuarioCreate(BaseModel):
    persona: PersonaBase
    cuenta: CuentaBase

class UsuarioResponse(BaseModel):
    id_persona: int
    nombre_completo: str
    correo_electronico: str
    telefono: str
    genero: Optional[str] = None
    activo: bool
    rol: str

class EstadoUpdate(BaseModel):
    activo: bool

class HistorialAcceso(BaseModel):
    id_acceso: int
    nombre_completo: str
    fecha: str
    resultado: str
    dispositivo: str
    foto_url: Optional[str] = None

class FiltrosHistorial(BaseModel):
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    resultado: Optional[str] = None
    nombre: Optional[str] = None

# ========== ENDPOINTS DE AUTENTICACIÓN ==========
@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("SELECT id_cuenta, contrasena_hash FROM cuentas WHERE nombre_usuario = :username"),
            {"username": user.username}
        )
        usuario = result.fetchone()

        if not usuario or not bcrypt.checkpw(
            user.password.encode('utf-8'),
            usuario.contrasena_hash.encode('utf-8')
        ):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        return {"status": "success", "user_id": usuario.id_cuenta}
        
    except Exception as e:
        logger.error(f"Error en login: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en el servidor")

@app.post("/registrar/", status_code=status.HTTP_201_CREATED)
def registrar_usuario(usuario: UsuarioCreate, db: Session = Depends(get_db)):
    try:
        # Verificar correo único
        if db.execute(
            text("SELECT 1 FROM personas WHERE correo_electronico = :email"),
            {"email": usuario.persona.correo_electronico}
        ).scalar():
            raise HTTPException(status_code=400, detail="El correo ya existe")

        # Hash de contraseña
        hashed_pw = bcrypt.hashpw(
            usuario.cuenta.password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        # Crear persona
        persona_id = db.execute(
            text("""
                INSERT INTO personas (
                    nombre, apellido_paterno, apellido_materno, 
                    telefono, correo_electronico, genero, activo
                ) VALUES (
                    :nombre, :apellido_p, :apellido_m,
                    :telefono, :email, :genero, TRUE
                ) RETURNING id_persona
            """),
            {
                "nombre": usuario.persona.nombre,
                "apellido_p": usuario.persona.apellido_paterno,
                "apellido_m": usuario.persona.apellido_materno,
                "telefono": usuario.persona.telefono,
                "email": usuario.persona.correo_electronico,
                "genero": usuario.persona.genero
            }
        ).scalar_one()

        # Crear cuenta
        db.execute(
            text("""
                INSERT INTO cuentas (
                    id_persona, id_rol, nombre_usuario, 
                    contrasena_hash, sal
                ) VALUES (
                    :id_persona, 
                    (SELECT id_rol FROM roles WHERE nombre = 'Usuario'),
                    :username, 
                    :password, 
                    ''
                )
            """),
            {
                "id_persona": persona_id,
                "username": usuario.persona.correo_electronico.split('@')[0],
                "password": hashed_pw
            }
        )

        db.commit()
        return {"status": "success", "id_persona": persona_id}

    except Exception as e:
        db.rollback()
        logger.error(f"Error en registro: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al registrar usuario")

# ========== ENDPOINTS DE USUARIOS ==========
@app.get("/personas/", response_model=List[UsuarioResponse])
def listar_usuarios(
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
        """)
        return db.execute(query, {"rol": rol, "activo": activo}).fetchall()
        
    except Exception as e:
        logger.error(f"Error listando usuarios: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al listar usuarios")

@app.get("/personas/{id_persona}", response_model=UsuarioResponse)
def obtener_usuario(id_persona: int, db: Session = Depends(get_db)):
    try:
        usuario = db.execute(
            text("""
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
                WHERE p.id_persona = :id AND r.nombre != 'Administrador'
            """),
            {"id": id_persona}
        ).fetchone()

        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
            
        return usuario
        
    except Exception as e:
        logger.error(f"Error obteniendo usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener usuario")

@app.put("/personas/{id_persona}/estado")
def actualizar_estado(
    id_persona: int,
    estado: EstadoUpdate,
    db: Session = Depends(get_db)
):
    try:
        # Verificar que existe y no es admin
        if not db.execute(
            text("""
                SELECT 1 FROM personas p
                JOIN cuentas c ON p.id_persona = c.id_persona 
                JOIN roles r ON c.id_rol = r.id_rol
                WHERE p.id_persona = :id AND r.nombre != 'Administrador'
            """),
            {"id": id_persona}
        ).scalar():
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        db.execute(
            text("UPDATE personas SET activo = :activo WHERE id_persona = :id"),
            {"activo": estado.activo, "id": id_persona}
        )
        db.commit()
        return {"status": "success", "activo": estado.activo}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error actualizando estado: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al actualizar estado")

# ========== ENDPOINTS DE HISTORIAL ==========
@app.get("/historial-accesos/", response_model=List[HistorialAcceso])
def obtener_historial(
    filtros: FiltrosHistorial = Depends(),
    limite: int = Query(20, gt=0, le=100),
    db: Session = Depends(get_db)
):
    try:
        params = {"limite": limite, "nombre": f"%{filtros.nombre}%" if filtros.nombre else "%"}
        
        query = text("""
            SELECT 
                ha.id_acceso,
                CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                TO_CHAR(ha.fecha, 'DD/MM/YYYY – HH:MI AM') as fecha,
                CASE WHEN ha.resultado = 'Éxito' THEN 'PERMITIDO' ELSE 'DENEGADO' END as resultado,
                COALESCE(d.nombre, 'Desconocido') as dispositivo,
                ha.foto_url
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            WHERE CONCAT(p.nombre, ' ', p.apellido_paterno) LIKE :nombre
        """)

        if filtros.fecha_inicio and filtros.fecha_fin:
            query = text(str(query) + " AND ha.fecha BETWEEN :fecha_inicio AND :fecha_fin")
            params.update({"fecha_inicio": filtros.fecha_inicio, "fecha_fin": filtros.fecha_fin})
        
        if filtros.resultado:
            query = text(str(query) + " AND ha.resultado = :resultado")
            params["resultado"] = 'Éxito' if filtros.resultado.upper() == 'PERMITIDO' else 'Fallo'

        query = text(str(query) + " ORDER BY ha.fecha DESC LIMIT :limite")
        
        return [
            dict(row) for row in db.execute(query, params).fetchall()
        ]
        
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener historial")

@app.get("/historial-accesos/{id_acceso}", response_model=HistorialAcceso)
def detalle_acceso(id_acceso: int, db: Session = Depends(get_db)):
    try:
        acceso = db.execute(
            text("""
                SELECT 
                    ha.id_acceso,
                    CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                    TO_CHAR(ha.fecha, 'DD/MM/YYYY – HH:MI AM') as fecha,
                    CASE WHEN ha.resultado = 'Éxito' THEN 'PERMITIDO' ELSE 'DENEGADO' END as resultado,
                    COALESCE(d.nombre, 'Desconocido') as dispositivo,
                    ha.foto_url
                FROM historial_accesos ha
                LEFT JOIN personas p ON ha.id_persona = p.id_persona
                LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
                WHERE ha.id_acceso = :id
            """),
            {"id": id_acceso}
        ).fetchone()

        if not acceso:
            raise HTTPException(status_code=404, detail="Registro no encontrado")
            
        return acceso
        
    except Exception as e:
        logger.error(f"Error obteniendo detalle: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener detalle")

# ========== UTILIDADES ==========
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}

@app.get("/generate-password/")
def generate_password(password: str):
    return {
        "hashed": bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    }
