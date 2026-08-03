# Egg Crack Detection Backend

Pure OpenCV backend for detecting dark cracks, bright light-leak cracks, and pale cracks on candled eggs.

## Main structure

```text
src/app/
├── api/main.py
├── core/config.py
├── detection/
│   ├── calibration.py
│   ├── config.py
│   ├── fuzzy.py
│   ├── models.py
│   ├── segmentation.py
│   ├── preprocessing.py
│   ├── thresholding.py
│   ├── paper_baseline.py
│   ├── components.py
│   ├── pipeline.py
│   └── rendering.py
├── repositories/db.py
├── schemas/detection.py
└── services/detector.py
```

## Crack detection

The detector uses egg segmentation, quality validation, lighting correction, CLAHE, separate dark and bright crack enhancement, hysteresis thresholding, morphology, skeleton measurements, component filtering, false-positive rejection, and multi-frame confirmation.

The separate paper baseline keeps the red-channel, green-channel, Gaussian blur, binary mask, morphology, edge, subtraction, and contour workflow from the supplied paper.

## Egg-size detection

Egg size now uses only calibrated egg width and height.

```text
Fixed camera at 4 inches
        ↓
Pixel-to-millimeter calibration
        ↓
OpenCV rotated egg width and height
        ↓
Mamdani fuzzy inference
        ↓
Small, medium, or large
```

When calibration exists, egg size uses width and height in millimeters. Without calibration, it uses Mamdani fuzzy classification from apparent width and height at the fixed 4-inch setup. Physical millimeter fields remain unavailable until calibration is completed.

The Mamdani system uses minimum for AND, maximum aggregation, and centroid defuzzification. Its width and height membership functions are in `src/app/detection/config.py`. The included values are starter ranges and should be tuned with manually measured eggs from the final setup.

## Required physical setup

- Camera lens fixed exactly 4 inches from the egg measurement plane
- Same camera, resolution, focus, zoom, and angle
- Same holder position
- Same dark box and candling light
- Calibration reference placed at the same plane as the egg

The `camera_distance_inches = 4.0` setting records the required setup. Real measurement comes from the saved `pixels_per_mm` calibration.

## Calibrate with a reference image

Place one known solid circle or square at the egg position. A 30 mm marker is supported by default.

```bat
.venv\Scripts\python.exe scripts\calibrate_camera.py reference.png --reference-width-mm 30 --shape circle
```

The profile is saved to:

```text
data/calibration.json
```

API calibration endpoints:

```text
GET    /calibration
POST   /calibration/reference
POST   /calibration/manual
DELETE /calibration
```

Reference calibration request:

```text
POST /calibration/reference
multipart/form-data:
  image: reference.png
  reference_width_mm: 30
  reference_shape: circle
```

Manual calibration request:

```json
{
  "reference_width_mm": 30,
  "reference_width_pixels": 300,
  "processed_width": 1280,
  "processed_height": 720
}
```

## Detection response fields

```json
{
  "egg_size": "medium",
  "egg_size_confidence": 0.79,
  "egg_width_pixels": 430.0,
  "egg_height_pixels": 570.0,
  "egg_width_mm": 43.0,
  "egg_height_mm": 57.0,
  "egg_measurement_valid": true,
  "camera_distance_inches": 4.0,
  "calibration_pixels_per_mm": 10.0
}
```

## Run

```bat
install_backend.bat
run_backend.bat
```

The API runs at:

```text
http://127.0.0.1:8756
```

## Test

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Evaluate real images

```bat
.venv\Scripts\python.exe scripts\evaluate_folder.py --cracked data\cracked --clean data\clean --output evaluation.json
```

Use real images from the final 4-inch camera, dark box, light, holder, and resolution. Synthetic tests verify the logic but do not replace physical calibration and validation.

## Camera orientation and overlay lines

Camera and uploaded images keep their native pixel orientation. The backend does not rotate, mirror, or horizontally flip captured frames. Egg borders, crack polygons, calibration contours, and component boxes are rendered at one pixel.

## Egg size without calibration

The backend no longer returns `unknown` only because `data/calibration.json` is missing. It uses Mamdani fuzzy classification from the detected egg height and width at the fixed 4-inch setup. The dimensions are normalized by the shorter frame side, so rotating the camera between portrait and landscape does not change the size class.

A saved calibration remains the preferred mode because it converts the dimensions to millimeters. Check `egg_size_mode`: `apparent_4_inch` is the automatic fallback, while `calibrated_mm` is the physically calibrated result.

## Whole visible egg detection area

Crack detection uses the complete visible egg surface and removes only the outermost one-pixel contour. The deeper inner mask is still used for focus scoring and image-quality checks. Cracks close to the edge are accepted when they are thin, long enough, and continue inward or follow a radial crack direction. The back side of the egg is not visible in one image and still requires rotating the egg.

## Full crack trace and multiple visible cracks

The detector keeps multiple independent strict crack anchors and uses narrow directional support to complete faint connected sections without accepting unrelated shell texture. The final crack mask now keeps the complete validated crack region instead of reducing it to a one-pixel centerline. The displayed polygon is still one pixel thick, but it follows the outer boundary of the full detected crack area.

## Camera preview orientation

Camera and uploaded images preserve the received pixel orientation by default. No automatic mirror or rotation is applied. Only set an orientation override when the camera plugin itself sends a mirrored or rotated encoded frame.

Example optional horizontal correction:

```powershell
$env:EGG_CAMERA_ORIENTATION_FIX="flip_horizontal"
.\run_backend.bat
```

Supported values are `none`, `flip_horizontal`, `flip_vertical`, `rotate_180`, `rotate_90_clockwise`, and `rotate_90_counterclockwise`.
