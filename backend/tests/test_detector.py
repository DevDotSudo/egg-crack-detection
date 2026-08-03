import base64
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.core.config import CONFIG
from app.detection.pipeline import EggCrackPipeline
from app.schemas.detection import DetectionResponse
from app.services.detector import (
    DetectionError,
    detect_camera_images_bytes,
    detect_image_bytes,
    score_camera_focus_image_bytes,
)


def candled_egg(
    crack: str | None = None,
    textured: bool = False,
    horizontal: bool = False,
) -> bytes:
    height, width = 540, 960
    yy, xx = np.mgrid[:height, :width]
    center_x, center_y = 480, 270
    radius_x, radius_y = (230, 185) if horizontal else (185, 230)
    normalized = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
    egg = normalized <= 1.0
    hotspot = np.exp(-(((xx - 515) / 190.0) ** 2 + ((yy - 255) / 250.0) ** 2))
    shell = np.clip(145 + hotspot * 76 - normalized * 20, 0, 255)
    image = np.full((height, width, 3), 4, dtype=np.uint8)
    image[:, :, 0][egg] = np.clip(shell[egg] * 0.42, 0, 255).astype(np.uint8)
    image[:, :, 1][egg] = np.clip(shell[egg] * 0.78, 0, 255).astype(np.uint8)
    image[:, :, 2][egg] = np.clip(shell[egg] * 1.06, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1.1, image.shape[:2]).astype(np.int16)
    for channel in range(3):
        values = image[:, :, channel].astype(np.int16)
        values[egg] += noise[egg]
        image[:, :, channel] = np.clip(values, 0, 255).astype(np.uint8)
    if textured:
        texture_rng = np.random.default_rng(19)
        for _ in range(220):
            while True:
                x = int(texture_rng.integers(center_x - 150, center_x + 151))
                y = int(texture_rng.integers(center_y - 185, center_y + 186))
                if normalized[y, x] < 0.72:
                    break
            length = int(texture_rng.integers(16, 27))
            angle = float(texture_rng.uniform(0, np.pi))
            dx = int(round(np.cos(angle) * length / 2.0))
            dy = int(round(np.sin(angle) * length / 2.0))
            darkening = int(texture_rng.integers(14, 27))
            color = tuple(max(int(value) - darkening, 0) for value in image[y, x])
            cv2.line(image, (x - dx, y - dy), (x + dx, y + dy), color, 1, cv2.LINE_8)
    points = np.array([
        [365, 205], [405, 222], [438, 211], [472, 246],
        [510, 235], [548, 274], [585, 263], [620, 302],
    ], dtype=np.int32)
    if crack == 'dark':
        cv2.polylines(image, [points], False, (34, 42, 50), 1, cv2.LINE_8)
    elif crack == 'faint_dark':
        for first, second in zip(points[:-1], points[1:]):
            middle = ((first + second) // 2).astype(int)
            color = tuple(max(int(value) - 12, 0) for value in image[middle[1], middle[0]])
            cv2.line(image, tuple(first), tuple(second), color, 1, cv2.LINE_8)
    elif crack == 'dark_on_texture':
        for first, second in zip(points[:-1], points[1:]):
            middle = ((first + second) // 2).astype(int)
            color = tuple(max(int(value) - 30, 0) for value in image[middle[1], middle[0]])
            cv2.line(image, tuple(first), tuple(second), color, 1, cv2.LINE_8)
    elif crack == 'bright':
        cv2.polylines(image, [points], False, (245, 252, 255), 1, cv2.LINE_8)
    return encode_png(image)


def modified_egg(kind: str) -> bytes:
    image = cv2.imdecode(
        np.frombuffer(candled_egg('dark' if kind == 'blur' else None), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if kind == 'yolk_arc':
        cv2.ellipse(image, (480, 285), (105, 72), 0, 190, 350, (42, 76, 105), 3, cv2.LINE_AA)
    elif kind == 'dark_texture':
        texture = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.ellipse(texture, (430, 245), (48, 27), 18, 0, 360, 1.0, -1)
        cv2.ellipse(texture, (525, 305), (38, 22), -24, 0, 360, 0.8, -1)
        texture = cv2.GaussianBlur(texture, (31, 31), 0)
        for channel in range(3):
            image[:, :, channel] = np.clip(
                image[:, :, channel].astype(np.float32) - texture * (42.0 + channel * 5.0),
                0,
                255,
            ).astype(np.uint8)
    elif kind == 'rim_texture':
        rng = np.random.default_rng(111)
        for _ in range(80):
            x = int(rng.integers(340, 620))
            y = int(rng.integers(360, 455))
            length = int(rng.integers(20, 45))
            angle = float(rng.normal(0.05, 0.22))
            dx = int(np.cos(angle) * length / 2.0)
            dy = int(np.sin(angle) * length / 2.0)
            cv2.line(image, (x - dx, y - dy), (x + dx, y + dy), (55, 75, 95), 1, cv2.LINE_AA)
    elif kind == 'shell_fiber':
        points = np.array([[575, 130], [584, 180], [578, 240], [587, 300], [580, 360], [590, 415]], dtype=np.int32)
        cv2.polylines(image, [points], False, (58, 78, 98), 2, cv2.LINE_AA)
    elif kind == 'pale_surface_crack':
        first = np.array([[292, 352], [323, 335], [358, 333]], dtype=np.int32)
        second = np.array([[365, 330], [398, 309]], dtype=np.int32)
        cv2.polylines(image, [first], False, (172, 184, 170), 2, cv2.LINE_AA)
        cv2.polylines(image, [second], False, (172, 184, 170), 2, cv2.LINE_AA)
    elif kind == 'pale_branching_surface_crack':
        branches = (
            np.array([[470, 180], [474, 225], [478, 270], [476, 315]], dtype=np.int32),
            np.array([[476, 315], [440, 330], [405, 342], [365, 360]], dtype=np.int32),
            np.array([[478, 270], [510, 292], [540, 325]], dtype=np.int32),
        )
        for branch in branches:
            cv2.polylines(image, [branch], False, (178, 190, 168), 2, cv2.LINE_AA)
    elif kind == 'pale_shell_arc':
        cv2.ellipse(image, (480, 395), (34, 15), 0, 200, 340, (178, 190, 168), 2, cv2.LINE_AA)
    elif kind == 'glare':
        cv2.line(image, (390, 150), (570, 390), (255, 255, 255), 22, cv2.LINE_AA)
    elif kind == 'blur':
        image = cv2.GaussianBlur(image, (11, 11), 0)
    return encode_png(image)


def transformed_egg(crack: str | None, x: float = 0.0, y: float = 0.0, angle: float = 0.0) -> bytes:
    image = cv2.imdecode(np.frombuffer(candled_egg(crack), dtype=np.uint8), cv2.IMREAD_COLOR)
    transform = cv2.getRotationMatrix2D((480.0, 270.0), angle, 1.0)
    transform[0, 2] += x
    transform[1, 2] += y
    image = cv2.warpAffine(
        image,
        transform,
        (960, 540),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(4, 4, 4),
    )
    return encode_png(image)


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise AssertionError('Image encoding failed')
    return encoded.tobytes()


def edge_cracked_egg(location: str, polarity: str) -> bytes:
    image = cv2.imdecode(np.frombuffer(candled_egg(), dtype=np.uint8), cv2.IMREAD_COLOR)
    paths = {
        'left': np.array([[300, 260], [315, 255], [330, 265], [350, 250], [375, 260], [400, 245]], dtype=np.int32),
        'top': np.array([[480, 44], [475, 65], [488, 85], [478, 110], [492, 135], [485, 165]], dtype=np.int32),
        'bottom': np.array([[480, 494], [474, 474], [487, 454], [478, 430], [490, 405], [482, 380]], dtype=np.int32),
    }
    color = (30, 38, 45) if polarity == 'dark' else (245, 252, 255)
    cv2.polylines(image, [paths[location]], False, color, 1, cv2.LINE_8)
    return encode_png(image)


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = replace(
            CONFIG,
            target_width=960,
            target_height=540,
            camera_orientation_fix='none',
            min_egg_width=60,
            min_egg_height=90,
            min_egg_pixels=4000,
            min_inner_pixels=2500,
        )

    def test_clean_egg_is_not_cracked(self) -> None:
        result = detect_image_bytes(candled_egg(), cfg=self.config)
        self.assertFalse(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 0)
        DetectionResponse.model_validate(result)

    def test_dark_and_bright_cracks_are_detected(self) -> None:
        dark = detect_image_bytes(candled_egg('dark'), cfg=self.config)
        bright = detect_image_bytes(candled_egg('bright'), cfg=self.config)
        self.assertTrue(dark['is_crack'], dark)
        self.assertTrue(bright['is_crack'], bright)
        self.assertEqual(dark['primary_detection_channel'], 'dark')
        self.assertEqual(bright['primary_detection_channel'], 'bright')

    def test_faint_dark_crack_is_detected(self) -> None:
        result = detect_image_bytes(candled_egg('faint_dark'), cfg=self.config)
        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['candidate_pixels'], 0)

    def test_texture_is_rejected_but_dominant_crack_survives(self) -> None:
        clean = detect_image_bytes(candled_egg(textured=True), cfg=self.config)
        cracked = detect_image_bytes(candled_egg('dark_on_texture', textured=True), cfg=self.config)
        self.assertFalse(clean['is_crack'], clean)
        self.assertTrue(cracked['is_crack'], cracked)
        self.assertTrue(cracked['dominant_crack_override'], cracked)
        self.assertEqual(cracked['candidate_components'], 1)

    def test_false_crack_shapes_are_rejected(self) -> None:
        for kind in ('yolk_arc', 'dark_texture', 'rim_texture', 'shell_fiber', 'pale_shell_arc'):
            with self.subTest(kind=kind):
                result = detect_image_bytes(modified_egg(kind), cfg=self.config)
                self.assertFalse(result['is_crack'], result)

    def test_pale_cracks_are_detected(self) -> None:
        for kind in ('pale_surface_crack', 'pale_branching_surface_crack'):
            with self.subTest(kind=kind):
                result = detect_image_bytes(modified_egg(kind), include_steps=True, cfg=self.config)
                self.assertTrue(result['is_crack'], result)
                self.assertEqual(result['primary_detection_channel'], 'bright')
                self.assertIn('paper_crack_mask', result['intermediate_steps'])
                self.assertIn('dark_crack_response', result['intermediate_steps'])
                self.assertIn('bright_crack_response', result['intermediate_steps'])

    def test_glare_and_blur_request_recapture(self) -> None:
        with self.assertRaisesRegex(DetectionError, 'glare'):
            detect_image_bytes(modified_egg('glare'), cfg=self.config)
        with self.assertRaisesRegex(DetectionError, 'blurry'):
            detect_image_bytes(modified_egg('blur'), cfg=self.config)

    def test_camera_consensus_accepts_stable_and_rejects_transient(self) -> None:
        stable = detect_camera_images_bytes(
            [
                transformed_egg('dark'),
                transformed_egg('dark', 8, -5, 1.1),
                transformed_egg(None, -4, 3, -0.8),
            ],
            cfg=self.config,
        )
        transient = detect_camera_images_bytes(
            [
                transformed_egg('dark'),
                transformed_egg(None, 6, -4),
                transformed_egg(None, -5, 3),
            ],
            cfg=self.config,
        )
        self.assertTrue(stable['is_crack'], stable)
        self.assertGreaterEqual(stable['crack_votes'], 2)
        self.assertFalse(transient['is_crack'], transient)
        self.assertEqual(transient['termination_reason'], 'multi_frame_disagreement')

    def test_focus_score_prefers_sharp_image(self) -> None:
        sharp_data = candled_egg('dark', textured=True)
        image = cv2.imdecode(np.frombuffer(sharp_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        blurred_data = encode_png(cv2.GaussianBlur(image, (15, 15), 0))
        sharp = score_camera_focus_image_bytes(sharp_data, self.config)
        blurred = score_camera_focus_image_bytes(blurred_data, self.config)
        self.assertGreater(sharp['focus_score'], blurred['focus_score'])

    def test_focus_score_targets_the_egg_not_the_background(self) -> None:
        sharp_data = candled_egg('dark', textured=True)
        sharp_image = cv2.imdecode(np.frombuffer(sharp_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        blurred_image = cv2.GaussianBlur(sharp_image, (17, 17), 0)
        yy, xx = np.mgrid[:sharp_image.shape[0], :sharp_image.shape[1]]
        egg_mask = ((xx - 480) / 185.0) ** 2 + ((yy - 270) / 230.0) ** 2 <= 1.0
        checker = (((xx // 3) + (yy // 3)) % 2) * 20 + 2
        for channel in range(3):
            blurred_image[:, :, channel][~egg_mask] = checker[~egg_mask].astype(np.uint8)
        sharp = score_camera_focus_image_bytes(sharp_data, self.config)
        blurred = score_camera_focus_image_bytes(encode_png(blurred_image), self.config)
        self.assertEqual(sharp['focus_region'], 'inner_egg')
        self.assertGreater(sharp['focus_score'], blurred['focus_score'])
        self.assertIn('texture_sharpness', sharp)
        self.assertIn('detail_variance', sharp)

    def test_cracks_are_detected_across_the_visible_egg(self) -> None:
        cases = (
            ('left', 'dark'),
            ('left', 'bright'),
            ('top', 'dark'),
            ('bottom', 'bright'),
        )
        for location, polarity in cases:
            with self.subTest(location=location, polarity=polarity):
                result = detect_image_bytes(edge_cracked_egg(location, polarity), include_steps=True, cfg=self.config)
                self.assertTrue(result['is_crack'], result)
                self.assertEqual(result['primary_detection_channel'], polarity)
                self.assertIn('whole_egg_detection_mask', result['intermediate_steps'])
                self.assertGreater(result['candidate_pixels'], 0)

    def test_real_bright_crack_survives_textured_shell_filter(self) -> None:
        fixture = Path(__file__).resolve().parent / 'fixtures' / 'real_bright_crack.png'
        result = detect_image_bytes(fixture.read_bytes(), cfg=self.config)
        self.assertTrue(result['is_crack'], result)
        self.assertEqual(result['primary_detection_channel'], 'bright')
        self.assertTrue(result['dominant_crack_override'], result)
        self.assertGreater(result['candidate_pixels'], 0)

    def test_real_long_crack_is_traced_in_full_without_mirroring(self) -> None:
        fixture = Path(__file__).resolve().parent / 'fixtures' / 'real_long_bright_crack.png'
        result = detect_image_bytes(fixture.read_bytes(), include_steps=True, cfg=self.config)
        self.assertTrue(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 1, result)
        location = result['crack_locations'][0]
        x, _, width, _ = location['bounding_box']
        source = cv2.imdecode(
            np.frombuffer(base64.b64decode(result['original_image_b64']), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertGreaterEqual(width, 150, result)
        self.assertGreater(x + width / 2.0, source.shape[1] / 2.0, result)
        self.assertIn('directional_ridge_extension', location['reasons'])
        self.assertIn('directional_ridge_extension_mask', result['intermediate_steps'])


    def test_user_bright_crack_stays_on_the_correct_side_without_texture_flooding(self) -> None:
        fixture = Path(__file__).resolve().parent / 'fixtures' / 'user_long_bright_crack.png'
        result = detect_image_bytes(fixture.read_bytes(), include_steps=True, cfg=self.config)
        self.assertTrue(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 1, result)
        x, _, width, height = result['crack_locations'][0]['bounding_box']
        source = cv2.imdecode(
            np.frombuffer(base64.b64decode(result['original_image_b64']), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertGreater(x + width / 2.0, source.shape[1] / 2.0, result)
        self.assertLessEqual(height, 32, result)
        self.assertGreater(result['candidate_pixels'], 500, result)
        self.assertLess(result['candidate_pixels'], 1600, result)
        self.assertNotIn('paper_guided_full_trace', result['crack_locations'][0]['reasons'])

    def test_multiple_visible_cracks_are_not_reduced_to_one_winner(self) -> None:
        image = cv2.imdecode(np.frombuffer(candled_egg('bright'), dtype=np.uint8), cv2.IMREAD_COLOR)
        second = np.array([[360, 365], [405, 340], [450, 360], [500, 335], [550, 355]], dtype=np.int32)
        cv2.polylines(image, [second], False, (245, 252, 255), 1, cv2.LINE_8)
        result = detect_image_bytes(encode_png(image), cfg=self.config)
        self.assertTrue(result['is_crack'], result)
        self.assertGreaterEqual(result['candidate_components'], 2, result)

    def test_final_mask_covers_the_validated_crack_area_not_only_its_skeleton(self) -> None:
        fixture = Path(__file__).resolve().parent / 'fixtures' / 'user_long_bright_crack.png'
        image = cv2.imdecode(np.frombuffer(fixture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        pipeline = EggCrackPipeline(self.config).detect(image)
        accepted = [component for component in pipeline.components if component.accepted]
        self.assertTrue(accepted)
        accepted_union = np.zeros_like(pipeline.crack_mask)
        for component in accepted:
            accepted_union = cv2.bitwise_or(accepted_union, component.mask)
        overlap = cv2.countNonZero(cv2.bitwise_and(pipeline.crack_mask, accepted_union))
        accepted_pixels = cv2.countNonZero(accepted_union)
        self.assertGreaterEqual(overlap, int(accepted_pixels * 0.98))
        self.assertGreater(cv2.countNonZero(pipeline.crack_mask), sum(component.skeleton_length for component in accepted))

    def test_original_resolution_is_not_upscaled(self) -> None:
        result = detect_image_bytes(candled_egg('dark'), cfg=CONFIG)
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(result['original_image_b64']), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(image.shape[:2], (540, 960))


if __name__ == '__main__':
    unittest.main()
