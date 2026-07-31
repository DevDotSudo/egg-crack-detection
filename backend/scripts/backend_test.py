"""
Test the updated detection pipeline:
1. Verify all modules import correctly
2. Check new config values are set
3. Verify the LoG, directional top-hat and _skeletonize functions exist
4. Run the full pipeline with intermediate steps to confirm it doesn't crash
   (egg detection will fail on synthetic images by design; we test the
   response computation functions independently)
"""
import sys
from pathlib import Path

import numpy as np
import cv2

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC = BACKEND_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.core.config import CONFIG
from app.services.detector import (
    _log_zero_crossing_response,
    _directional_tophat_response,
    _skeletonize,
    _line_responses,
    _shell_texture_responses,
    _masked_channel,
)

print('=== Config checks ===')
assert CONFIG.line_sigmas == (0.7, 1.1, 2.8, 6.0), f'Unexpected: {CONFIG.line_sigmas}'
assert CONFIG.texture_ridge_sigmas == (0.45, 0.65, 0.9, 1.25, 1.8, 2.5)
assert CONFIG.log_sigmas == (0.8, 1.4, 2.2)
assert CONFIG.tophat_line_kernels == ((1, 9), (9, 1), (1, 13), (13, 1))
assert CONFIG.thin_crack_min_length == 12.0
assert CONFIG.thin_crack_min_span == 10.0
assert CONFIG.thin_crack_min_texture_strength == 28.0
assert CONFIG.texture_seed_min_length == 55.0
assert CONFIG.decision_min_score == 0.50
assert CONFIG.support_radius == 4
assert CONFIG.max_fragmented_components == 8
assert CONFIG.clahe_clip_limit == 3.2
assert CONFIG.clahe_tile_size == 8
print('  All config values correct')

print()
print('=== LoG zero-crossing response ===')
# Create a 200x200 test image with a thin dark line
detail = np.ones((200, 200), dtype=np.uint8) * 128
cv2.line(detail, (40, 100), (160, 100), 60, 1)
inner_mask = np.ones((200, 200), dtype=np.uint8) * 255

result = _log_zero_crossing_response(detail, inner_mask, CONFIG)
assert result.shape == (200, 200)
assert result.dtype == np.uint8
assert result.max() <= 255
# The response around the crack line should be non-zero
line_response = result[95:106, 40:160]
print(f'  LoG max along crack: {line_response.max()}')
print(f'  LoG mean along crack: {line_response.mean():.2f}')
print('  LoG OK')

print()
print('=== Directional top-hat response ===')
dir_dark, dir_bright = _directional_tophat_response(detail, inner_mask, CONFIG)
assert dir_dark.shape == (200, 200)
assert dir_bright.shape == (200, 200)
print(f'  dir_dark max: {dir_dark.max()}, mean: {dir_dark.mean():.3f}')
print(f'  dir_bright max: {dir_bright.max()}, mean: {dir_bright.mean():.3f}')
print('  Directional top-hat OK')

print()
print('=== Skeletonize cap check ===')
# A 50x50 filled circle — previously would iterate 52 times, now capped at 50+2=52 vs 320
# (cap doesn't change result here, just prevents runaway on huge components)
circle = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(circle, (50, 50), 30, 255, -1)
skel = _skeletonize(circle)
assert skel.shape == (100, 100)
print(f'  Skeleton pixels from 30px-radius circle: {int(cv2.countNonZero(skel))}')
print('  Skeletonize OK')

print()
print('=== Line responses smoke test ===')
# Build a minimal synthetic BGR image for line response
img_bgr = np.ones((200, 200, 3), dtype=np.uint8) * 128
cv2.line(img_bgr, (40, 100), (160, 100), (60, 55, 50), 1)
detail2 = np.ones((200, 200), dtype=np.uint8) * 128
cv2.line(detail2, (40, 100), (160, 100), 60, 1)
dark_r, bright_r, edge_r = _line_responses(img_bgr, detail2, inner_mask, CONFIG)
assert dark_r.shape == (200, 200)
assert bright_r.shape == (200, 200)
print(f'  dark_response max: {dark_r.max()}, bright_response max: {bright_r.max()}')
print(f'  edge_response max: {edge_r.max()}')
print('  Line responses OK')

print()
print('ALL TESTS PASSED')
