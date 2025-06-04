from fastapi import FastAPI, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime
import logging
from fastapi.middleware.cors import CORSMiddleware
from database import get_db
import json

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

class ResumenAccesos(BaseModel):
    total_entradas: int
    total_retrasos: int
    total_salidas: int
    total_reportes: int
    usuarios_activos: int
    ultima_actualizacion: datetime

class FiltroHistorial(BaseModel):
    nombre_usuario: Optional[str] = None
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None
    estado_registro: Optional[str] = None

class UsuarioCompleto(BaseModel):
    id_persona: int
    nombre: str
    apellido_paterno: str
    apellido_materno: Optional[str]
    telefono: str
    correo_electronico: str
    activo: bool
    nombre_usuario: str
    rol: str
    hora_entrada: Optional[str]
    hora_salida: Optional[str]
    accesos: List[dict]
    total_accesos: int

class FiltroReportes(BaseModel):
    estado: Optional[str] = None
    limit: int = 5
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

@app.get("/resumen/", response_model=ResumenAccesos)
def obtener_resumen(db: Session = Depends(get_db)):
    """Endpoint para obtener un resumen general del sistema"""
    try:
        # Obtener total de entradas
        query_entradas = text("""
            SELECT COUNT(*) 
            FROM historial_accesos 
            WHERE estado_registro LIKE 'ENTRADA%'
        """)
        total_entradas = db.execute(query_entradas).scalar_one()

        # Obtener total de retrasos
        query_retrasos = text("""
            SELECT COUNT(*) 
            FROM historial_accesos 
            WHERE estado_registro LIKE 'RETRASO%'
        """)
        total_retrasos = db.execute(query_retrasos).scalar_one()

        # Obtener total de salidas
        query_salidas = text("""
            SELECT COUNT(*) 
            FROM historial_accesos 
            WHERE estado_registro LIKE 'SALIDA%'
        """)
        total_salidas = db.execute(query_salidas).scalar_one()

        # Obtener total de reportes
        query_reportes = text("""
            SELECT COUNT(*) 
            FROM reportes
        """)
        total_reportes = db.execute(query_reportes).scalar_one()

        # Obtener usuarios activos
        query_usuarios = text("""
            SELECT COUNT(*) 
            FROM personas 
            WHERE activo = TRUE
        """)
        usuarios_activos = db.execute(query_usuarios).scalar_one()

        return {
            "total_entradas": total_entradas,
            "total_retrasos": total_retrasos,
            "total_salidas": total_salidas,
            "total_reportes": total_reportes,
            "usuarios_activos": usuarios_activos,
            "ultima_actualizacion": datetime.now()
        }

    except Exception as e:
        logger.error(f"Error al obtener resumen: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al generar el resumen"
        )

@app.post("/historial/filtrar/")
def filtrar_historial(filtro: FiltroHistorial, db: Session = Depends(get_db)):
    """Endpoint para filtrar el historial de accesos"""
    try:
        query = text("""
            SELECT ha.*, p.nombre, p.apellido_paterno, p.apellido_materno, c.nombre_usuario
            FROM historial_accesos ha
            JOIN personas p ON ha.id_persona = p.id_persona
            JOIN cuentas c ON p.id_persona = c.id_persona
            WHERE 1=1
        """)
        params = {}

        # Aplicar filtros
        if filtro.nombre_usuario:
            query = text(f"{query.text} AND c.nombre_usuario LIKE :nombre_usuario")
            params["nombre_usuario"] = f"%{filtro.nombre_usuario}%"
        
        if filtro.fecha_inicio:
            query = text(f"{query.text} AND ha.fecha >= :fecha_inicio")
            params["fecha_inicio"] = filtro.fecha_inicio
        
        if filtro.fecha_fin:
            query = text(f"{query.text} AND ha.fecha <= :fecha_fin")
            params["fecha_fin"] = filtro.fecha_fin
        
        if filtro.estado_registro:
            query = text(f"{query.text} AND ha.estado_registro = :estado_registro")
            params["estado_registro"] = filtro.estado_registro

        query = text(f"{query.text} ORDER BY ha.fecha DESC")
        result = db.execute(query, params)
        
        historial = []
        for row in result:
            historial.append({
                "id_acceso": row.id_acceso,
                "nombre_completo": f"{row.nombre} {row.apellido_paterno} {row.apellido_materno or ''}".strip(),
                "nombre_usuario": row.nombre_usuario,
                "fecha": row.fecha,
                "resultado": row.resultado,
                "estado_registro": row.estado_registro,
                "dispositivo": row.id_dispositivo,
                "confianza": row.confianza
            })

        return {"historial": historial}

    except Exception as e:
        logger.error(f"Error al filtrar historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al filtrar el historial"
        )

@app.get("/usuarios/buscar/", response_model=List[UsuarioCompleto])
def buscar_usuarios(nombre: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """Endpoint para buscar usuarios por nombre y mostrar información completa"""
    try:
        # Buscar personas por nombre
        query_personas = text("""
            SELECT p.*, c.nombre_usuario, r.nombre as rol
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE p.nombre LIKE :nombre OR p.apellido_paterno LIKE :nombre
        """)
        result_personas = db.execute(query_personas, {"nombre": f"%{nombre}%"})
        
        usuarios = []
        for persona in result_personas:
            # Obtener horario
            query_horario = text("""
                SELECT hora_entrada, hora_salida 
                FROM horarios_persona 
                WHERE id_persona = :id_persona
            """)
            horario = db.execute(query_horario, {"id_persona": persona.id_persona}).fetchone()
            
            # Obtener accesos
            query_accesos = text("""
                SELECT ha.*, d.nombre as dispositivo_nombre
                FROM historial_accesos ha
                LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
                WHERE ha.id_persona = :id_persona
                ORDER BY ha.fecha DESC
            """)
            accesos_result = db.execute(query_accesos, {"id_persona": persona.id_persona})
            
            accesos = []
            for acceso in accesos_result:
                accesos.append({
                    "fecha": acceso.fecha,
                    "dispositivo": acceso.dispositivo_nombre,
                    "resultado": acceso.resultado,
                    "estado_registro": acceso.estado_registro,
                    "confianza": acceso.confianza
                })
            
            usuarios.append({
                "id_persona": persona.id_persona,
                "nombre": persona.nombre,
                "apellido_paterno": persona.apellido_paterno,
                "apellido_materno": persona.apellido_materno,
                "telefono": persona.telefono,
                "correo_electronico": persona.correo_electronico,
                "activo": persona.activo,
                "nombre_usuario": persona.nombre_usuario,
                "rol": persona.rol,
                "hora_entrada": str(horario.hora_entrada) if horario else None,
                "hora_salida": str(horario.hora_salida) if horario else None,
                "accesos": accesos,
                "total_accesos": len(accesos)
            })

        return usuarios

    except Exception as e:
        logger.error(f"Error al buscar usuarios: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al buscar usuarios"
        )

@app.get("/reportes/", response_model=List[dict])
def obtener_reportes(filtro: FiltroReportes = Depends(), db: Session = Depends(get_db)):
    """Endpoint para obtener reportes filtrados por estado"""
    try:
        query = text("""
            SELECT r.*, p.nombre, p.apellido_paterno
            FROM reportes r
            LEFT JOIN historial_accesos ha ON r.id_acceso_relacionado = ha.id_acceso
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            WHERE 1=1
        """)
        params = {}
        
        if filtro.estado:
            query = text(f"{query.text} AND r.estado = :estado")
            params["estado"] = filtro.estado
        
        query = text(f"{query.text} ORDER BY r.fecha_generacion DESC LIMIT :limit")
        params["limit"] = filtro.limit
        
        result = db.execute(query, params)
        
        reportes = []
        for row in result:
            reportes.append({
                "id_reporte": row.id_reporte,
                "titulo": row.titulo,
                "descripcion": row.descripcion,
                "tipo_reporte": row.tipo_reporte,
                "severidad": row.severidad,
                "estado": row.estado,
                "fecha_generacion": row.fecha_generacion,
                "persona": f"{row.nombre or ''} {row.apellido_paterno or ''}".strip() or None,
                "evidencias": row.evidencias
            })
        
        return reportes

    except Exception as e:
        logger.error(f"Error al obtener reportes: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al obtener reportes"
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
