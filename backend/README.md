# Egg Crack Detection Backend

The backend owns all image processing and detection. It does not search for cameras, open webcams, stream previews, or capture frames. Flutter only detects cameras, shows the preview, captures the original image bytes, and sends those bytes to the API.

## Structure

- `src/app/api/` contains FastAPI app wiring and endpoint handlers.
- `src/app/core/` contains shared configuration.
- `src/app/services/` contains detection and image-processing services.
- `src/app/repositories/` contains persistence adapters such as history storage.
- `src/app/schemas/` contains API request and response models.
- `tests/` contains unit and contract tests.
- `scripts/` contains development diagnostics and smoke-test helpers.
- `data/` contains local runtime history.

## Detection flow

1. Receive the untouched captured image.
2. Preserve the captured pixel orientation without mirroring or forced rotation.
3. Detect and validate the egg.
4. Ignore the background and shell rim with an inner egg mask.
5. Correct uneven candling light and enhance local shell detail.
6. Enhance local contrast with CLAHE, then extract dark and bright directional morphological line maps.
7. Filter connected components by span, skeleton length, thickness, and line shape.
8. Reject smooth yolk-like curves, broad glare bands, fragmented pores, and normal shell markings.
9. Reject blurry, overexposed, glare-heavy, underexposed, or low-contrast captures and ask for a new image.
10. Keep native camera resolution and only downscale images larger than the configured limit.
11. Classify egg size with fuzzy logic using area, width, and length.
12. For camera captures, require a spatially matching crack trace in multiple frames.
13. Return the overlay, crack details, texture metrics, image-quality metrics, fuzzy egg size, and processing time.

## Endpoints

- `GET /health`
- `POST /detect`
- `POST /detect/camera`
- `POST /detect/camera/multi`
- `POST /detect/batch`
- `GET /history`
- `POST /history`
- `DELETE /history`
- `GET /history/{id}`
- `DELETE /history/{id}`
- `GET /reports/export`

There are no backend camera endpoints.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
.\run_backend.bat
```

The server runs at `http://127.0.0.1:8756`.

## Test

```powershell
python -m unittest discover -s tests -v
```

Egg size is an apparent size estimate. Keep the camera distance and position fixed for consistent labels. Physical grading requires calibration or a known reference.

## Image-quality behavior

The detector returns HTTP 422 with a simple recapture message when the image is too blurry, too dark, overexposed, low contrast, or covered by broad flashlight glare. It does not save those rejected images to history.

Returned quality fields for accepted images:

- `image_quality_score`
- `image_sharpness`
- `image_detail_variance`
- `image_saturated_ratio`
- `image_glare_ratio`
- `image_dynamic_range`
- `requires_recapture`
- `quality_message`

## Paper-based crack detection pipeline

The detector now follows Purahong et al. (2022): resize for faster processing, split red and green channels, apply an 11x11 Gaussian blur to the red channel, create a binary egg mask, multiply that mask with the green channel, perform edge detection, apply the 3x3 cross morphology kernel, remove the egg boundary, and filter crack contours by area and length. Detected crack regions are returned as filled red polygons with a thin red outline, while the egg keeps a thin green border.
