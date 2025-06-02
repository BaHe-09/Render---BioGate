from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from PIL import Image
import io
from keras_facenet import FaceNet
from typing import List, Optional
from sqlalchemy.orm import Session
from database import get_db
from sqlalchemy import text
import logging
from datetime import datetime
import uuid
import os
import csv


app = FastAPI(
    title="Person Registration API",
    description="API para registrar personas con sus vectores de identificación facial",
    version="1.0.0"
)

# Configuración de CORS (importante para el frontend)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe esto a tus dominios
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
IMAGE_UPLOAD_DIR = "uploaded_images"
os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)

# Cargar el modelo FaceNet
facenet = FaceNet()

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convierte la imagen al formato requerido por FaceNet"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

@app.get("/health-check")
async def health_check():
    return {"status": "ok", "message": "API de registro de personas operativa"}

@app.post("/register_person", response_model=dict)
async def register_person(
    nombre: str = Form(...),
    apellido_paterno: str = Form(...),
    apellido_materno: Optional[str] = Form(None),
    telefono: Optional[str] = Form(None),
    correo_electronico: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint para registrar una nueva persona con sus vectores de identificación facial
    
    Parámetros:
    - nombre: Nombre de la persona (requerido)
    - apellido_paterno: Primer apellido (requerido)
    - apellido_materno: Segundo apellido (opcional)
    - telefono: Número de teléfono (opcional)
    - correo_electronico: Correo electrónico (opcional, debe ser único)
    - images: Lista de imágenes del rostro (al menos 1 requerida)
    
    Retorna:
    - Información de la persona registrada
    - Número de vectores faciales almacenados
    """
    # Validar que se hayan subido imágenes
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="Se requiere al menos una imagen del rostro")

    try:
        # Registrar la persona en la tabla personas
        persona_query = text("""
            INSERT INTO personas 
            (nombre, apellido_paterno, apellido_materno, telefono, correo_electronico)
            VALUES 
            (:nombre, :apellido_paterno, :apellido_materno, :telefono, :correo_electronico)
            RETURNING id_persona
        """)
        
        result = db.execute(persona_query, {
            "nombre": nombre,
            "apellido_paterno": apellido_paterno,
            "apellido_materno": apellido_materno,
            "telefono": telefono,
            "correo_electronico": correo_electronico
        })
        
        id_persona = result.fetchone()[0]
        vectors_created = 0
        saved_images = []
        
        # Procesar cada imagen
        for img_file in images:
            if not img_file.content_type.startswith('image/'):
                logger.warning(f"Archivo {img_file.filename} no es una imagen, omitiendo")
                continue
                
            try:
                # Leer y procesar imagen
                contents = await img_file.read()
                img = Image.open(io.BytesIO(contents))
                img_array = preprocess_image(img)
                
                # Obtener embeddings
                detections = facenet.embeddings([img_array])
                
                if len(detections) == 0:
                    logger.warning(f"No se detectaron rostros en {img_file.filename}")
                    continue
                
                embedding = detections[0].tolist()
                
                # Guardar imagen en el sistema de archivos
                unique_filename = f"{uuid.uuid4()}.jpg"
                image_path = os.path.join(IMAGE_UPLOAD_DIR, unique_filename)
                img.save(image_path)
                saved_images.append(image_path)
                
                # Registrar el vector en la base de datos
                vector_query = text("""
                    INSERT INTO vectores_identificacion 
                    (id_persona, vector, dispositivo_registro)
                    VALUES 
                    (:id_persona, :vector, :dispositivo)
                """)
                
                db.execute(vector_query, {
                    "id_persona": id_persona,
                    "vector": "[" + ",".join(map(str, embedding)) + "]",
                    "dispositivo": "API Registration"
                })
                
                vectors_created += 1
                
            except Exception as e:
                logger.error(f"Error al procesar imagen {img_file.filename}: {str(e)}")
                continue
        
        # Confirmar todos los cambios en la base de datos
        db.commit()
        
        # Si no se pudo crear ningún vector, eliminar la persona registrada
        if vectors_created == 0:
            db.execute(text("DELETE FROM personas WHERE id_persona = :id"), {"id": id_persona})
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="No se pudo detectar ningún rostro en las imágenes proporcionadas"
            )
        
        return {
            "message": "Persona registrada exitosamente",
            "id_persona": id_persona,
            "nombre_completo": f"{nombre} {apellido_paterno} {apellido_materno or ''}".strip(),
            "vectors_created": vectors_created,
            "saved_images": saved_images
        }
    
    except Exception as e:
        db.rollback()
        logger.exception("Error en el registro de persona")
        
        # Manejar error de correo duplicado
        if "duplicate key value violates unique constraint" in str(e) and "correo_electronico" in str(e):
            raise HTTPException(
                status_code=400,
                detail="El correo electrónico ya está registrado"
            )
        
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al registrar persona: {str(e)}"
        )

# Endpoint para exportar reportes a CSV
@app.get("/reportes/exportar-csv", response_class=StreamingResponse)
async def exportar_reportes_csv(
    tipo_reporte: Optional[str] = Query(None, description="Filtrar por tipo de reporte"),
    estado: Optional[str] = Query(None, description="Filtrar por estado del reporte"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    Exporta reportes a formato CSV con filtros opcionales.
    """
    try:
        # Construir la consulta SQL base
        query = text("""
            SELECT 
                id_reporte, titulo, descripcion, tipo_reporte,
                severidad, estado, fecha_generacion, fecha_cierre,
                id_acceso_relacionado, id_dispositivo
            FROM reportes
            WHERE 1=1
        """)
        
        params = {}
        
        # Aplicar filtros
        if tipo_reporte:
            query = text(f"{query.text} AND tipo_reporte = :tipo_reporte")
            params["tipo_reporte"] = tipo_reporte
            
        if estado:
            query = text(f"{query.text} AND estado = :estado")
            params["estado"] = estado
            
        if fecha_inicio:
            query = text(f"{query.text} AND fecha_generacion >= :fecha_inicio")
            params["fecha_inicio"] = fecha_inicio
            
        if fecha_fin:
            query = text(f"{query.text} AND fecha_generacion <= :fecha_fin")
            params["fecha_fin"] = fecha_fin + " 23:59:59"  # Incluir todo el día
        
        # Ordenar por fecha de generación descendente
        query = text(f"{query.text} ORDER BY fecha_generacion DESC")
        
        # Ejecutar consulta
        result = db.execute(query, params)
        reportes = result.fetchall()
        
        # Crear CSV en memoria
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)

        # Escribir encabezados
        headers = [
            "ID Reporte", "Título", "Descripción", "Tipo de Reporte",
            "Severidad", "Estado", "Fecha Generación", "Fecha Cierre",
            "ID Acceso Relacionado", "ID Dispositivo"
        ]
        writer.writerow(headers)

        # Escribir datos
        for reporte in reportes:
            writer.writerow([
                reporte.id_reporte,
                reporte.titulo,
                reporte.descripcion,
                reporte.tipo_reporte,
                reporte.severidad,
                reporte.estado,
                reporte.fecha_generacion.strftime("%Y-%m-%d %H:%M:%S") if reporte.fecha_generacion else "",
                reporte.fecha_cierre.strftime("%Y-%m-%d %H:%M:%S") if reporte.fecha_cierre else "",
                reporte.id_acceso_relacionado or "",
                reporte.id_dispositivo or ""
            ])

        output.seek(0)
        
        # Configurar respuesta para descarga
        filename = f"reportes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar el archivo CSV: {str(e)}"
        )

# Endpoint para exportar historial de accesos a CSV
@app.get("/accesos/exportar-csv", response_class=StreamingResponse)
async def exportar_accesos_csv(
    estado_registro: Optional[str] = Query(None, description="Filtrar por tipo de registro (ENTRADA/SALIDA)"),
    resultado: Optional[str] = Query(None, description="Filtrar por resultado (Exito/Fallo)"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    Exporta historial de accesos a formato CSV con filtros opcionales.
    Los caracteres especiales se reemplazan por equivalentes sin acentos.
    """
    try:
        # Consulta SQL (sin cambios)
        query = text("""
            SELECT 
                ha.id_acceso, 
                CONCAT(p.nombre, ' ', p.apellido_paterno, ' ', COALESCE(p.apellido_materno, '')) AS nombre_completo,
                d.nombre AS dispositivo,
                ha.fecha,
                ha.estado_registro,
                ha.resultado,
                ha.confianza,
                ha.horas_extras,
                ha.es_dia_laboral,
                ha.razon,
                ha.foto_url
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            WHERE 1=1
        """)
        
        # ... (resto del código de filtros igual)

        # Función para eliminar acentos
        def remove_accents(input_str):
            replacements = {
                'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
                'ñ': 'n', 'Ñ': 'N'
            }
            for orig, repl in replacements.items():
                input_str = input_str.replace(orig, repl)
            return input_str

        # Crear CSV en memoria con codificación UTF-8
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)

        # Encabezados sin acentos
        headers = [
            "ID Acceso", "Nombre Completo", "Dispositivo", "Fecha y Hora",
            "Tipo de Registro", "Resultado", "Nivel de Confianza", "Horas Extras",
            "Dia Laboral", "Razon", "URL Foto"
        ]
        writer.writerow(headers)

        for acceso in accesos:
            # Procesar campos de texto para eliminar acentos
            nombre = remove_accents(acceso.nombre_completo) if acceso.nombre_completo else ""
            dispositivo = remove_accents(acceso.dispositivo) if acceso.dispositivo else ""
            estado = remove_accents(acceso.estado_registro) if acceso.estado_registro else ""
            resultado = remove_accents(acceso.resultado) if acceso.resultado else ""
            razon = remove_accents(acceso.razon) if acceso.razon else ""

            writer.writerow([
                acceso.id_acceso,
                nombre,
                dispositivo,
                acceso.fecha.strftime("%Y-%m-%d %H:%M:%S"),
                estado,
                resultado,
                f"{acceso.confianza:.2f}" if acceso.confianza is not None else "",
                f"{acceso.horas_extras:.2f}" if acceso.horas_extras else "0.00",
                "Si" if acceso.es_dia_laboral else "No",
                razon,
                acceso.foto_url or ""
            ])

        output.seek(0)
        
        # Configurar respuesta con codificación UTF-8
        filename = f"accesos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([output.getvalue().encode('utf-8')]),  # Codificar a UTF-8
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar el archivo CSV: {str(e)}"
        )
        
@app.get("/", include_in_schema=False)
def root():
    return {"message": "API de registro de personas con identificación facial"}
