import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.core.config import CONFIG
from app.detection.calibration import CalibrationStore, CameraCalibrator
from app.services.detector import _correct_camera_orientation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--reference-width-mm', type=float, default=30.0)
    parser.add_argument('--shape', choices=('circle', 'square', 'bar'), default='circle')
    parser.add_argument('--overlay', default='calibration_overlay.png')
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit('Could not read the calibration image')
    image = _correct_camera_orientation(image, CONFIG)
    profile, overlay = CameraCalibrator(CONFIG).image_profile(
        image,
        args.reference_width_mm,
        args.shape,
    )
    CalibrationStore(ROOT / 'data' / 'calibration.json').save(profile)
    cv2.imwrite(args.overlay, overlay)
    print(json.dumps(profile.to_dict(), indent=2))


if __name__ == '__main__':
    main()
