import csv
import io
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.core.config import CONFIG
from app.repositories.history_store import HistoryStore
from app.schemas.detection import DetectionResponse, HistorySaveRequest
from app.services.detector import (
    DetectionError,
    detect_camera_image_bytes,
    detect_camera_images_bytes,
    detect_image_bytes,
    score_camera_focus_image_bytes,
)

BASE_DIR = Path(__file__).resolve().parents[3]
store = HistoryStore(BASE_DIR / 'data' / 'history.json')
app = FastAPI(title='Egg Crack Detection API', version='1.5.0')
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
        'version': '1.5',
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


@app.post('/detect', response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    include_intermediate_steps: bool = Form(False),
    save_history: bool = Form(True),
):
    try:
        filename, data = await _read_image_upload(image)
        result = detect_image_bytes(
            data,
            include_intermediate_steps,
            CONFIG,
        )
        result['source_name'] = filename
        if save_history:
            store.add(result)
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
            result = detect_image_bytes(data, include_intermediate_steps)
            result['source_name'] = filename
            if save_history:
                store.add(result)
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


@app.get('/history')
def history(limit: int = Query(100, ge=1, le=1000)):
    return {'items': store.list(limit)}


@app.post('/history')
def save_history(request: HistorySaveRequest):
    item = request.result.model_dump()
    item['source_name'] = request.source_name.strip() or 'camera'
    created = store.add(item)
    return {
        'saved': True,
        'created': created,
        'id': request.result.id,
    }


@app.delete('/history')
def clear_history():
    return {'deleted': store.clear()}


@app.get('/history/{item_id}')
def history_item(item_id: str):
    item = store.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail='History item not found')
    return item


@app.delete('/history/{item_id}')
def delete_history_item(item_id: str):
    if not store.delete(item_id):
        raise HTTPException(status_code=404, detail='History item not found')
    return {'deleted': True}


@app.get('/reports/export')
def export_report():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['id', 'source_name', 'is_crack', 'egg_size', 'egg_size_confidence', 'egg_size_score', 'egg_area_ratio', 'egg_width_pixels', 'egg_length_pixels', 'egg_width_ratio', 'egg_length_ratio', 'crack_size', 'crack_size_confidence', 'detection_iterations', 'search_iterations', 'termination_reason', 'thin_crack_detected', 'thin_crack_score', 'shell_texture_score', 'shell_texture_uniformity', 'texture_anomaly_ratio', 'texture_candidate_pixels', 'image_quality_score', 'image_sharpness', 'image_detail_variance', 'image_saturated_ratio', 'image_glare_ratio', 'image_dynamic_range', 'requires_recapture', 'quality_message', 'confidence', 'area_ratio', 'contour_length', 'processing_time_ms', 'timestamp'])
    writer.writeheader()
    for item in store.list(1000):
        writer.writerow({key: item.get(key, '') for key in writer.fieldnames})
    data = io.BytesIO(output.getvalue().encode('utf-8'))
    return StreamingResponse(data, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=egg_crack_report.csv'})
