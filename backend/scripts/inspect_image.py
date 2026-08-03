import argparse
import base64
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.services.detector import detect_image_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('inspection_output'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = detect_image_bytes(args.image.read_bytes(), include_steps=True)
    image_fields = ('original_image_b64', 'overlay_image_b64', 'crack_mask_b64')
    for field in image_fields:
        suffix = '.jpg' if field == 'original_image_b64' else '.png'
        (args.output / f'{field}{suffix}').write_bytes(base64.b64decode(result[field]))
    for name, value in (result.get('intermediate_steps') or {}).items():
        (args.output / f'{name}.png').write_bytes(base64.b64decode(value))
    metadata = {
        key: value
        for key, value in result.items()
        if key not in {*image_fields, 'intermediate_steps'} and not key.startswith('_internal_')
    }
    (args.output / 'result.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({
        'is_crack': result['is_crack'],
        'confidence': result['confidence'],
        'components': result['candidate_components'],
        'output': str(args.output.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
