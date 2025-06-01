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

class DetalleDispositivo(BaseModel):
    nombre: str
    ubicacion: str

class DetallesAcceso(BaseModel):
    hora_entrada: str
    hora_salida: str

class DetalleAccesoCompleto(BaseModel):
    id_acceso: int
    nombre_completo: str
    fecha: str
    horario: str
    dispositivo: DetalleDispositivo
    estatus: str
    detalles_acceso: DetallesAcceso
    dias_laborales: Optional[str] = None
    nivel_confianza: Optional[float] = None
    estado_registro: Optional[str] = None
    es_dia_laboral: bool
    razon: str
    foto_url: Optional[str] = None

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

class ReporteBase(BaseModel):
    titulo: str
    descripcion: Optional[str] = None
    tipo_reporte: str
    severidad: Optional[str] = "Media"
    estado: Optional[str] = "Abierto"
    id_acceso_relacionado: Optional[int] = None
    id_dispositivo: Optional[int] = None
    etiquetas: Optional[dict] = None
    evidencias: Optional[List[str]] = None

class ReporteCreate(ReporteBase):
    pass

class ReporteResponse(ReporteBase):
    id_reporte: int
    fecha_generacion: datetime
    fecha_cierre: Optional[datetime] = None
    generado_por: Optional[int] = None
    
    class Config:
        from_attributes = True

class ReporteUpdate(BaseModel):
    titulo: Optional[str] = None
    descripcion: Optional[str] = None
    estado: Optional[str] = None
    severidad: Optional[str] = None
    etiquetas: Optional[dict] = None
    evidencias: Optional[List[str]] = None

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
        # Consulta base
        base_query = """
            SELECT 
                ha.id_acceso,
                CASE 
                    WHEN p.nombre IS NULL THEN 'DESCONOCIDO'
                    ELSE CONCAT(p.nombre, ' ', p.apellido_paterno, ' ', p.apellido_materno)
                END as nombre_completo,
                TO_CHAR(ha.fecha, 'DD/MM/YYYY – HH:MI AM') as fecha,
                CASE 
                    WHEN ha.resultado = 'Éxito' THEN 'PERMITIDO'
                    ELSE 'DENEGADO'
                END as resultado,
                COALESCE(d.ubicacion, 'Desconocida') as dispositivo,
                ha.foto_url
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            WHERE 1=1
        """
        
        query_params = {
            "limite": limite,
            "nombre": f"%{filtros.nombre}%" if filtros.nombre else "%"
        }

        # Construir condiciones dinámicas
        conditions = []
        
        # Filtro por nombre
        conditions.append("""
            AND (
                CASE 
                    WHEN p.nombre IS NULL THEN 'DESCONOCIDO'
                    ELSE CONCAT(p.nombre, ' ', p.apellido_paterno, ' ', p.apellido_materno)
                END LIKE :nombre
            )
        """)
        
        # Filtros de fecha
        if filtros.fecha_inicio and filtros.fecha_fin:
            conditions.append("AND ha.fecha BETWEEN :fecha_inicio AND :fecha_fin")
            query_params.update({
                "fecha_inicio": filtros.fecha_inicio,
                "fecha_fin": filtros.fecha_fin
            })
        
        # Filtro por resultado
        if filtros.resultado:
            if filtros.resultado.upper() == 'PERMITIDO':
                conditions.append("AND ha.resultado = 'Éxito'")
            elif filtros.resultado.upper() == 'DENEGADO':
                conditions.append("AND ha.resultado != 'Éxito'")

        # Construir la consulta final
        final_query = base_query + "\n".join(conditions) + "\nORDER BY ha.fecha DESC LIMIT :limite"
        
        result = db.execute(text(final_query), query_params)
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
        
@app.get("/historial-accesos/{id_acceso}", response_model=DetalleAccesoCompleto)
def obtener_detalle_acceso(id_acceso: int, db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT 
                ha.id_acceso,
                CONCAT(p.nombre, ' ', p.apellido_paterno, ' ', COALESCE(p.apellido_materno, '')) as nombre_completo,
                TO_CHAR(ha.fecha, 'DD/MM/YYYY') as fecha,
                TO_CHAR(ha.fecha, 'HH:MI AM') as horario,
                hp.hora_entrada,
                hp.hora_salida,
                hp.dias_laborales,
                CASE 
                    WHEN ha.resultado = 'Éxito' THEN 'PERMITIDO'
                    ELSE 'DENEGADO'
                END as estatus,
                COALESCE(d.nombre, 'Desconocido') as nombre_dispositivo,
                COALESCE(d.ubicacion, 'Desconocida') as ubicacion_dispositivo,
                ha.confianza,
                ha.estado_registro,
                ha.es_dia_laboral,
                COALESCE(ha.razon, 'N/A') as razon,
                ha.foto_url
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            LEFT JOIN horarios_persona hp ON ha.id_persona = hp.id_persona
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
            "horario": acceso.horario,
            "dispositivo": {
                "nombre": acceso.nombre_dispositivo,
                "ubicacion": acceso.ubicacion_dispositivo
            },
            "estatus": acceso.estatus,
            "detalles_acceso": {
                "hora_entrada": str(acceso.hora_entrada) if acceso.hora_entrada else "N/A",
                "hora_salida": str(acceso.hora_salida) if acceso.hora_salida else "N/A"
            },
            "dias_laborales": acceso.dias_laborales,
            "nivel_confianza": acceso.confianza * 100 if acceso.confianza else None,
            "estado_registro": acceso.estado_registro,
            "es_dia_laboral": acceso.es_dia_laboral,
            "razon": acceso.razon,
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

@app.post("/reportes/", response_model=ReporteResponse, status_code=status.HTTP_201_CREATED)
def crear_reporte(
    reporte: ReporteCreate, 
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user)  # Asume que tienes un sistema de autenticación
):
    try:
        query = text("""
            INSERT INTO reportes (
                titulo, descripcion, tipo_reporte, severidad, estado,
                generado_por, id_acceso_relacionado, id_dispositivo, etiquetas, evidencias
            )
            VALUES (
                :titulo, :descripcion, :tipo_reporte, :severidad, :estado,
                :generado_por, :id_acceso_relacionado, :id_dispositivo, :etiquetas, :evidencias
            )
            RETURNING id_reporte, fecha_generacion
        """)
        
        result = db.execute(query, {
            "titulo": reporte.titulo,
            "descripcion": reporte.descripcion,
            "tipo_reporte": reporte.tipo_reporte,
            "severidad": reporte.severidad,
            "estado": reporte.estado,
            "generado_por": current_user_id,
            "id_acceso_relacionado": reporte.id_acceso_relacionado,
            "id_dispositivo": reporte.id_dispositivo,
            "etiquetas": json.dumps(reporte.etiquetas) if reporte.etiquetas else None,
            "evidencias": reporte.evidencias
        })
        
        nuevo_reporte = result.fetchone()
        db.commit()
        
        return {
            "id_reporte": nuevo_reporte.id_reporte,
            **reporte.dict(),
            "fecha_generacion": nuevo_reporte.fecha_generacion,
            "generado_por": current_user_id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear reporte: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al crear el reporte"
        )

@app.get("/reportes/", response_model=List[ReporteResponse])
def listar_reportes(
    estado: Optional[str] = None,
    tipo: Optional[str] = None,
    severidad: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        base_query = """
            SELECT id_reporte, titulo, descripcion, tipo_reporte, severidad, estado,
                   fecha_generacion, fecha_cierre, generado_por, 
                   id_acceso_relacionado, id_dispositivo, etiquetas, evidencias
            FROM reportes
            WHERE 1=1
        """
        
        conditions = []
        params = {}
        
        if estado:
            conditions.append("AND estado = :estado")
            params["estado"] = estado
            
        if tipo:
            conditions.append("AND tipo_reporte = :tipo")
            params["tipo"] = tipo
            
        if severidad:
            conditions.append("AND severidad = :severidad")
            params["severidad"] = severidad
            
        query = text(base_query + " ".join(conditions) + " ORDER BY fecha_generacion DESC")
        result = db.execute(query, params)
        
        return [
            {
                "id_reporte": r.id_reporte,
                "titulo": r.titulo,
                "descripcion": r.descripcion,
                "tipo_reporte": r.tipo_reporte,
                "severidad": r.severidad,
                "estado": r.estado,
                "fecha_generacion": r.fecha_generacion,
                "fecha_cierre": r.fecha_cierre,
                "generado_por": r.generado_por,
                "id_acceso_relacionado": r.id_acceso_relacionado,
                "id_dispositivo": r.id_dispositivo,
                "etiquetas": r.etiquetas,
                "evidencias": r.evidencias
            }
            for r in result.fetchall()
        ]
        
    except Exception as e:
        logger.error(f"Error al listar reportes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener los reportes"
        )

@app.put("/reportes/{id_reporte}", response_model=ReporteResponse)
def actualizar_reporte(
    id_reporte: int,
    reporte: ReporteUpdate,
    db: Session = Depends(get_db)
):
    try:
        # Verificar si existe
        existe = db.execute(
            text("SELECT 1 FROM reportes WHERE id_reporte = :id"),
            {"id": id_reporte}
        ).scalar()
        
        if not existe:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reporte no encontrado"
            )
            
        # Construir la consulta dinámica
        update_fields = []
        params = {"id": id_reporte}
        
        if reporte.titulo:
            update_fields.append("titulo = :titulo")
            params["titulo"] = reporte.titulo
            
        if reporte.descripcion:
            update_fields.append("descripcion = :descripcion")
            params["descripcion"] = reporte.descripcion
            
        if reporte.estado:
            update_fields.append("estado = :estado")
            params["estado"] = reporte.estado
            
            if reporte.estado in ("Resuelto", "Cerrado"):
                update_fields.append("fecha_cierre = CURRENT_TIMESTAMP")
                
        if reporte.severidad:
            update_fields.append("severidad = :severidad")
            params["severidad"] = reporte.severidad
            
        if reporte.etiquetas:
            update_fields.append("etiquetas = :etiquetas")
            params["etiquetas"] = json.dumps(reporte.etiquetas)
            
        if reporte.evidencias:
            update_fields.append("evidencias = :evidencias")
            params["evidencias"] = reporte.evidencias
            
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se proporcionaron campos para actualizar"
            )
            
        query = text(f"""
            UPDATE reportes
            SET {", ".join(update_fields)}
            WHERE id_reporte = :id
            RETURNING *
        """)
        
        result = db.execute(query, params)
        updated = result.fetchone()
        db.commit()
        
        return {
            "id_reporte": updated.id_reporte,
            "titulo": updated.titulo,
            "descripcion": updated.descripcion,
            "tipo_reporte": updated.tipo_reporte,
            "severidad": updated.severidad,
            "estado": updated.estado,
            "fecha_generacion": updated.fecha_generacion,
            "fecha_cierre": updated.fecha_cierre,
            "generado_por": updated.generado_por,
            "id_acceso_relacionado": updated.id_acceso_relacionado,
            "id_dispositivo": updated.id_dispositivo,
            "etiquetas": updated.etiquetas,
            "evidencias": updated.evidencias
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar reporte: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al actualizar el reporte"
        )
        
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
