import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.services.detector import DetectionError, detect_image_bytes


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob('*') if path.suffix.lower() in EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cracked', type=Path, required=True)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('evaluation.json'))
    args = parser.parse_args()
    records = []
    counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'rejected': 0}
    for expected, folder in ((True, args.cracked), (False, args.clean)):
        for path in images(folder):
            try:
                result = detect_image_bytes(path.read_bytes())
                predicted = bool(result['is_crack'])
                key = 'tp' if expected and predicted else 'fn' if expected else 'fp' if predicted else 'tn'
                counts[key] += 1
                records.append({
                    'file': str(path),
                    'expected_crack': expected,
                    'predicted_crack': predicted,
                    'confidence': result['confidence'],
                    'channel': result['primary_detection_channel'],
                    'components': result['candidate_components'],
                })
            except DetectionError as exc:
                counts['rejected'] += 1
                records.append({'file': str(path), 'expected_crack': expected, 'error': str(exc)})
    precision = counts['tp'] / max(counts['tp'] + counts['fp'], 1)
    recall = counts['tp'] / max(counts['tp'] + counts['fn'], 1)
    specificity = counts['tn'] / max(counts['tn'] + counts['fp'], 1)
    report = {
        'counts': counts,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'records': records,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('counts', 'precision', 'recall', 'specificity')}, indent=2))


if __name__ == '__main__':
    main()
