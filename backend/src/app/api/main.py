import csv
import io
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import CONFIG
from app.core.paths import runtime_root
from app.detection.calibration import CalibrationError, CameraCalibrator
from app.repositories.db import DetectionDB
from app.schemas.detection import (
    CalibrationProfileResponse,
    DetectionResponse,
    HistorySaveRequest,
    ManualCalibrationRequest,
)
from app.services.detector import (
    CALIBRATION_STORE,
    DetectionError,
    _correct_camera_orientation,
    _decode_input_image,
    _encode_image,
    detect_camera_image_bytes,
    detect_camera_images_bytes,
    detect_image_bytes,
    score_camera_focus_image_bytes,
)

BASE_DIR = runtime_root()
db = DetectionDB(BASE_DIR / 'data' / 'detections.db')

# Remove stale JSON history file if it still exists on disk.
_legacy_json = BASE_DIR / 'data' / 'history.json'
if _legacy_json.exists():
    try:
        _legacy_json.unlink()
    except OSError:
        pass

app = FastAPI(title='Egg Crack Detection API', version='2.3.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'engine': 'opencv-image-processing-only',
        'camera_owner': 'flutter-frontend',
        'version': '2.3',
        'camera_distance_inches': CONFIG.calibration.camera_distance_inches,
        'egg_size_calibrated': CALIBRATION_STORE.load() is not None,
    }


@app.get('/calibration', response_model=CalibrationProfileResponse)
def get_calibration():
    profile = CALIBRATION_STORE.load()
    if profile is None:
        return {
            'calibrated': False,
            'required_camera_distance_inches': CONFIG.calibration.camera_distance_inches,
            'profile': None,
            'message': 'Calibrate the fixed 4-inch camera setup before using egg-size classification',
        }
    return {
        'calibrated': True,
        'required_camera_distance_inches': CONFIG.calibration.camera_distance_inches,
        'profile': profile.to_dict(),
        'message': 'The egg-size scale is calibrated for the fixed 4-inch camera setup',
    }


@app.post('/calibration/manual', response_model=CalibrationProfileResponse)
def save_manual_calibration(request: ManualCalibrationRequest):
    try:
        profile = CameraCalibrator(CONFIG).manual_profile(
            request.reference_width_mm,
            request.reference_width_pixels,
            request.processed_width,
            request.processed_height,
        )
        CALIBRATION_STORE.save(profile)
        return {
            'calibrated': True,
            'required_camera_distance_inches': CONFIG.calibration.camera_distance_inches,
            'profile': profile.to_dict(),
            'message': 'Manual pixel-to-millimeter calibration was saved',
        }
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/calibration/reference', response_model=CalibrationProfileResponse)
async def save_reference_calibration(
    image: UploadFile = File(...),
    reference_width_mm: float = Form(30.0),
    reference_shape: str = Form('circle'),
):
    try:
        _, data = await _read_image_upload(image)
        source = _correct_camera_orientation(_decode_input_image(data), CONFIG)
        profile, overlay = CameraCalibrator(CONFIG).image_profile(
            source,
            reference_width_mm,
            reference_shape,
        )
        CALIBRATION_STORE.save(profile)
        return {
            'calibrated': True,
            'required_camera_distance_inches': CONFIG.calibration.camera_distance_inches,
            'profile': profile.to_dict(),
            'message': 'Reference-image pixel-to-millimeter calibration was saved',
            'overlay_image_b64': _encode_image(overlay, '.png'),
        }
    except (CalibrationError, DetectionError, cv2.error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete('/calibration')
def clear_calibration():
    return {
        'deleted': CALIBRATION_STORE.clear(),
        'required_camera_distance_inches': CONFIG.calibration.camera_distance_inches,
    }


async def _read_image_upload(image: UploadFile) -> tuple[str, bytes]:
    filename = image.filename or 'uploaded-image'
    extension = Path(filename).suffix.lower()
    if extension not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        raise DetectionError('Use a JPG, JPEG, PNG, BMP, or WEBP image')
    data = await image.read()
    if not data:
        raise DetectionError('The uploaded image is empty')
    if len(data) > 20 * 1024 * 1024:
        raise DetectionError('The uploaded image must be 20 MB or smaller')
    return filename, data


# ---------------------------------------------------------------------------
# Detection endpoints
# ---------------------------------------------------------------------------

@app.post('/detect', response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    include_intermediate_steps: bool = Form(False),
    save_history: bool = Form(True),
):
    try:
        filename, data = await _read_image_upload(image)
        result = detect_image_bytes(data, include_intermediate_steps, CONFIG)
        result['source_name'] = filename
        if save_history:
            db.add(result)
        return result
    except (DetectionError, cv2.error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/detect/batch')
async def detect_batch(
    images: list[UploadFile] = File(...),
    include_intermediate_steps: bool = Form(False),
    save_history: bool = Form(True),
):
    results = []
    for image in images:
        try:
            filename, data = await _read_image_upload(image)
            result = detect_image_bytes(data, include_intermediate_steps, CONFIG)
            result['source_name'] = filename
            if save_history:
                db.add(result)
            results.append({'filename': filename, 'ok': True, 'result': result})
        except (DetectionError, cv2.error) as exc:
            results.append({'filename': image.filename or 'uploaded-image', 'ok': False, 'error': str(exc)})
    return {'results': results, 'total': len(results)}


@app.post('/detect/camera', response_model=DetectionResponse)
async def detect_camera(
    image: UploadFile = File(...),
    include_intermediate_steps: bool = Form(False),
):
    try:
        _, data = await _read_image_upload(image)
        return detect_camera_image_bytes(data, include_intermediate_steps, CONFIG)
    except (DetectionError, cv2.error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/detect/camera/multi', response_model=DetectionResponse)
async def detect_camera_multi(
    images: list[UploadFile] = File(...),
    include_intermediate_steps: bool = Form(False),
):
    try:
        frames = []
        for image in images:
            _, data = await _read_image_upload(image)
            frames.append(data)
        return detect_camera_images_bytes(frames, include_intermediate_steps, CONFIG)
    except (DetectionError, cv2.error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/focus/score')
async def score_camera_focus(image: UploadFile = File(...)):
    try:
        _, data = await _read_image_upload(image)
        return score_camera_focus_image_bytes(data, CONFIG)
    except (DetectionError, cv2.error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# History endpoints — backed by SQLite (same paths kept for client compat)
# ---------------------------------------------------------------------------

@app.get('/history')
def history(limit: int = Query(100, ge=1, le=1000)):
    return {'items': db.list(limit)}


@app.post('/history')
def save_history(request: HistorySaveRequest):
    item = request.result.model_dump()
    item['source_name'] = request.source_name.strip() or 'camera'
    created = db.add(item)
    return {'saved': True, 'created': created, 'id': request.result.id}


@app.delete('/history')
def clear_history():
    return {'deleted': db.clear()}


@app.get('/history/{item_id}')
def history_item(item_id: str):
    item = db.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail='History item not found')
    return item


@app.delete('/history/{item_id}')
def delete_history_item(item_id: str):
    if not db.delete(item_id):
        raise HTTPException(status_code=404, detail='History item not found')
    return {'deleted': True}


# ---------------------------------------------------------------------------
# Detections endpoints — same DB, explicit image-serving routes
# ---------------------------------------------------------------------------

@app.get('/detections')
def list_detections(limit: int = Query(100, ge=1, le=1000)):
    return {'items': db.list(limit), 'limit': limit}


@app.get('/detections/{item_id}')
def get_detection(item_id: str):
    item = db.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail='Detection not found')
    return item


@app.get('/detections/{item_id}/original')
def get_detection_original_image(item_id: str):
    original_path, _ = db.get_image_paths(item_id)
    if not original_path:
        raise HTTPException(status_code=404, detail='Original image not found')
    p = Path(original_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail='Original image file missing from disk')
    return FileResponse(str(p), media_type='image/jpeg', filename=p.name)


@app.get('/detections/{item_id}/overlay')
def get_detection_overlay_image(item_id: str):
    _, overlay_path = db.get_image_paths(item_id)
    if not overlay_path:
        raise HTTPException(status_code=404, detail='Overlay image not found')
    p = Path(overlay_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail='Overlay image file missing from disk')
    return FileResponse(str(p), media_type='image/png', filename=p.name)


@app.delete('/detections/{item_id}')
def delete_detection(item_id: str):
    if not db.delete(item_id):
        raise HTTPException(status_code=404, detail='Detection not found')
    return {'deleted': True, 'id': item_id}


@app.delete('/detections')
def clear_detections():
    return {'deleted': db.clear()}


# ---------------------------------------------------------------------------
# CSV export — from SQLite
# ---------------------------------------------------------------------------

@app.get('/reports/export')
def export_report():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'id', 'source_name', 'timestamp', 'is_crack',
        'egg_size', 'egg_size_confidence', 'egg_size_score',
        'egg_area_ratio', 'egg_width_pixels', 'egg_height_pixels', 'egg_length_pixels',
        'egg_width_mm', 'egg_height_mm', 'egg_width_ratio', 'egg_length_ratio',
        'egg_measurement_valid', 'egg_measurement_message',
        'camera_distance_inches', 'calibration_pixels_per_mm',
        'crack_size', 'crack_size_confidence', 'crack_size_score',
        'detection_iterations', 'search_iterations', 'termination_reason',
        'thin_crack_detected', 'thin_crack_score',
        'shell_texture_score', 'shell_texture_uniformity',
        'texture_anomaly_ratio', 'texture_candidate_pixels',
        'image_quality_score', 'image_sharpness', 'image_detail_variance',
        'image_saturated_ratio', 'image_glare_ratio', 'image_dynamic_range',
        'requires_recapture', 'quality_message',
        'confidence', 'area_ratio', 'contour_length', 'processing_time_ms',
        'original_image_path', 'overlay_image_path',
    ])
    writer.writeheader()
    for item in db.list(1000):
        writer.writerow({key: item.get(key, '') for key in writer.fieldnames})
    data = io.BytesIO(output.getvalue().encode('utf-8'))
    return StreamingResponse(
        data,
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=egg_crack_report.csv'},
    )
