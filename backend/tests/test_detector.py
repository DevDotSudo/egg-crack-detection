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
from app.services.detector import (
    CrackComponent,
    DetectionError,
    _fuzzy_egg_size,
    _dominant_fragment_group,
    _is_dominant_crack_component,
    detect_camera_images_bytes,
    detect_camera_image_bytes,
    detect_image_bytes,
    score_camera_focus_image_bytes,
)
from app.schemas.detection import DetectionResponse


def _candled_egg(
    crack: str | None = None,
    textured: bool = False,
    horizontal: bool = False,
) -> bytes:
    height, width = 540, 960
    yy, xx = np.mgrid[:height, :width]
    center_x, center_y = 480, 270
    radius_x, radius_y = (230, 185) if horizontal else (185, 230)
    normalized = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
    )
    egg = normalized <= 1.0

    hotspot = np.exp(-(
        ((xx - 515) / 190.0) ** 2
        + ((yy - 255) / 250.0) ** 2
    ))
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
            color = tuple(
                max(int(value) - darkening, 0) for value in image[y, x]
            )
            cv2.line(
                image,
                (x - dx, y - dy),
                (x + dx, y + dy),
                color,
                1,
                cv2.LINE_8,
            )

    points = np.array([
        [365, 205], [405, 222], [438, 211], [472, 246],
        [510, 235], [548, 274], [585, 263], [620, 302],
    ], dtype=np.int32)
    if crack == 'dark':
        cv2.polylines(image, [points], False, (34, 42, 50), 1, cv2.LINE_8)
    elif crack == 'faint_dark':
        # Only ~12 grey levels below the shell — a challenging case for dark
        # crack detection, still well below the 14-27 level texture noise.
        for i in range(len(points) - 1):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + 1])
            mid_y = (pt1[1] + pt2[1]) // 2
            mid_x = (pt1[0] + pt2[0]) // 2
            base = image[mid_y, mid_x].astype(int)
            color = tuple(max(int(v) - 12, 0) for v in base)
            cv2.line(image, pt1, pt2, color, 1, cv2.LINE_8)
    elif crack == 'dark_on_texture':
        # 30 grey levels below shell — a moderate dark crack on a noisy surface
        for i in range(len(points) - 1):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + 1])
            mid_y = (pt1[1] + pt2[1]) // 2
            mid_x = (pt1[0] + pt2[0]) // 2
            base = image[mid_y, mid_x].astype(int)
            color = tuple(max(int(v) - 30, 0) for v in base)
            cv2.line(image, pt1, pt2, color, 1, cv2.LINE_8)
    elif crack == 'subtle':
        cv2.polylines(image, [points], False, (75, 135, 185), 1, cv2.LINE_8)
    elif crack == 'bright':
        cv2.polylines(image, [points], False, (245, 252, 255), 1, cv2.LINE_8)

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise AssertionError('Could not encode synthetic test image')
    return encoded.tobytes()


def _modified_candled_egg(kind: str) -> bytes:
    image = cv2.imdecode(
        np.frombuffer(_candled_egg('dark' if kind == 'blur' else None), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise AssertionError('Could not create stress-test image')

    if kind == 'yolk_arc':
        cv2.ellipse(
            image,
            (480, 285),
            (105, 72),
            0,
            190,
            350,
            (42, 76, 105),
            3,
            cv2.LINE_AA,
        )
    elif kind == 'dark_texture':
        # Broad, softly shaded shell marks are texture, not cracks. Their
        # outlines must not be promoted to hairlines by the dark channel.
        texture = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.ellipse(texture, (430, 245), (48, 27), 18, 0, 360, 1.0, -1)
        cv2.ellipse(texture, (525, 305), (38, 22), -24, 0, 360, 0.8, -1)
        cv2.circle(texture, (505, 205), 19, 0.65, -1)
        texture = cv2.GaussianBlur(texture, (31, 31), 0)
        for channel in range(3):
            values = image[:, :, channel].astype(np.float32)
            values -= texture * (42.0 + channel * 5.0)
            image[:, :, channel] = np.clip(values, 0, 255).astype(np.uint8)
    elif kind == 'rim_texture':
        # Short, repeated shell-grain strokes near the lower rim are not a
        # single fracture, even though the full-egg scan must include them.
        texture_rng = np.random.default_rng(111)
        for _ in range(80):
            x = int(texture_rng.integers(340, 620))
            y = int(texture_rng.integers(360, 455))
            length = int(texture_rng.integers(20, 45))
            angle = float(texture_rng.normal(0.05, 0.22))
            dx = int(np.cos(angle) * length / 2.0)
            dy = int(np.sin(angle) * length / 2.0)
            cv2.line(
                image,
                (x - dx, y - dy),
                (x + dx, y + dy),
                (55, 75, 95),
                1,
                cv2.LINE_AA,
            )
    elif kind == 'shell_fiber':
        # A smooth, slightly wider shell fiber should not be treated as a
        # crack simply because it forms one long darker ridge.
        points = np.array([
            [575, 130],
            [584, 180],
            [578, 240],
            [587, 300],
            [580, 360],
            [590, 415],
        ], dtype=np.int32)
        cv2.polylines(image, [points], False, (58, 78, 98), 2, cv2.LINE_AA)
    elif kind == 'pale_surface_crack':
        # A whitish shell fracture under candling can be brighter and less
        # saturated than the surrounding shell instead of dark.
        first = np.array([
            [292, 352],
            [323, 335],
            [358, 333],
        ], dtype=np.int32)
        second = np.array([
            [365, 330],
            [398, 309],
        ], dtype=np.int32)
        cv2.polylines(image, [first], False, (172, 184, 170), 2, cv2.LINE_AA)
        cv2.polylines(image, [second], False, (172, 184, 170), 2, cv2.LINE_AA)
    elif kind == 'glare':
        cv2.line(
            image,
            (390, 150),
            (570, 390),
            (255, 255, 255),
            22,
            cv2.LINE_AA,
        )
    elif kind == 'blur':
        image = cv2.GaussianBlur(image, (11, 11), 0)
    else:
        raise AssertionError(f'Unknown stress-test kind: {kind}')

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise AssertionError('Could not encode stress-test image')
    return encoded.tobytes()


def _peripheral_candled_egg() -> bytes:
    image = cv2.imdecode(
        np.frombuffer(_candled_egg(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise AssertionError('Could not create peripheral crack test image')

    points = np.array([
        [395, 402],
        [430, 425],
        [474, 414],
        [526, 438],
        [575, 416],
    ], dtype=np.int32)
    cv2.polylines(image, [points], False, (34, 42, 50), 1, cv2.LINE_8)

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise AssertionError('Could not encode peripheral crack image')
    return encoded.tobytes()


def _transformed_candled_egg(
    crack: str | None,
    *,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    angle: float = 0.0,
    brightness: int = 0,
) -> bytes:
    image = cv2.imdecode(
        np.frombuffer(_candled_egg(crack), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise AssertionError('Could not create transformed camera frame')
    transform = cv2.getRotationMatrix2D((480.0, 270.0), angle, 1.0)
    transform[0, 2] += translate_x
    transform[1, 2] += translate_y
    image = cv2.warpAffine(
        image,
        transform,
        (960, 540),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(4, 4, 4),
    )
    if brightness:
        image = np.clip(
            image.astype(np.int16) + brightness,
            0,
            255,
        ).astype(np.uint8)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise AssertionError('Could not encode transformed camera frame')
    return encoded.tobytes()


def _double_resolution(data: bytes) -> bytes:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError('Could not decode image for resolution test')
    resized = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode('.png', resized)
    if not ok:
        raise AssertionError('Could not encode resolution test image')
    return encoded.tobytes()


def _decode_overlay(result: dict) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(result['overlay_image_b64']), dtype=np.uint8)
    overlay = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if overlay is None:
        raise AssertionError('Could not decode detector overlay')
    return overlay


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = replace(
            CONFIG,
            target_width=960,
            target_height=540,
            min_egg_width=60,
            min_egg_height=90,
            min_egg_pixels=4000,
            min_inner_pixels=2500,
        )

    def test_camera_iteratively_extracts_until_no_crack_remains(self) -> None:
        result = detect_camera_image_bytes(
            _candled_egg('dark'),
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertEqual(result['sample_count'], 1, result)
        self.assertGreaterEqual(result['detection_iterations'], 1, result)
        self.assertEqual(
            len(result['crack_locations']),
            result['detection_iterations'],
        )
        self.assertGreater(
            result['search_iterations'],
            result['detection_iterations'],
        )
        self.assertEqual(result['termination_reason'], 'no_more_cracks')
        self.assertTrue(result['crack_mask_b64'])
        DetectionResponse.model_validate(result)

    def test_focus_score_prefers_sharp_illuminated_egg(self) -> None:
        sharp = _candled_egg('dark', textured=True)
        sharp_image = cv2.imdecode(
            np.frombuffer(sharp, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(sharp_image)
        blurred_image = cv2.GaussianBlur(sharp_image, (15, 15), 0)
        ok, blurred = cv2.imencode('.png', blurred_image)
        self.assertTrue(ok)

        sharp_score = score_camera_focus_image_bytes(sharp, self.config)
        blurred_score = score_camera_focus_image_bytes(
            blurred.tobytes(),
            self.config,
        )

        self.assertTrue(sharp_score['egg_detected'])
        self.assertGreater(
            sharp_score['focus_score'],
            blurred_score['focus_score'],
        )

    def test_camera_keeps_two_separate_crack_locations(self) -> None:
        image = cv2.imdecode(
            np.frombuffer(_candled_egg('dark'), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(image)
        second = np.array([
            [430, 330], [445, 350], [438, 370], [455, 390], [448, 410],
        ], dtype=np.int32)
        cv2.polylines(image, [second], False, (30, 38, 45), 1, cv2.LINE_8)
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)

        result = detect_camera_image_bytes(encoded.tobytes(), cfg=self.config)
        self.assertTrue(result['is_crack'], result)
        self.assertEqual(result['detection_iterations'], 2, result)
        self.assertEqual(len(result['crack_locations']), 2, result)
        first_box = result['crack_locations'][0]['bounding_box']
        second_box = result['crack_locations'][1]['bounding_box']
        self.assertNotEqual(first_box, second_box)
        DetectionResponse.model_validate(result)

    def test_camera_requires_a_spatially_stable_trace(self) -> None:
        result = detect_camera_images_bytes(
            [_candled_egg('dark'), _candled_egg('dark'), _candled_egg()],
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertEqual(result['sample_count'], 3, result)
        self.assertGreaterEqual(result['crack_votes'], 2, result)
        self.assertGreater(result['decision_consistency'], 0.5, result)
        DetectionResponse.model_validate(result)

    def test_registered_camera_consensus_survives_small_motion(self) -> None:
        result = detect_camera_images_bytes(
            [
                _transformed_candled_egg('dark'),
                _transformed_candled_egg(
                    'dark', translate_x=9, translate_y=-6, angle=1.4,
                    brightness=4,
                ),
                _transformed_candled_egg(
                    'dark', translate_x=-7, translate_y=5, angle=-1.1,
                    brightness=-3,
                ),
            ],
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertGreaterEqual(result['crack_votes'], 2, result)
        self.assertNotIn('_internal_crack_mask', result)
        DetectionResponse.model_validate(result)

    def test_camera_rejects_a_one_frame_transient(self) -> None:
        result = detect_camera_images_bytes(
            [
                _transformed_candled_egg('dark'),
                _transformed_candled_egg(None, translate_x=6, translate_y=-4),
                _transformed_candled_egg(None, translate_x=-5, translate_y=3),
            ],
            cfg=self.config,
        )
        self.assertFalse(result['is_crack'], result)
        self.assertEqual(result['crack_votes'], 0, result)
        self.assertEqual(result['termination_reason'], 'multi_frame_disagreement')
        DetectionResponse.model_validate(result)

    def test_clean_candled_egg_is_not_cracked(self) -> None:
        result = detect_image_bytes(_candled_egg(), cfg=self.config)
        overlay = _decode_overlay(result)
        red_pixels = np.count_nonzero(
            (overlay[:, :, 2] > 230)
            & (overlay[:, :, 1] < 60)
            & (overlay[:, :, 0] < 60)
        )
        green_pixels = np.count_nonzero(
            (overlay[:, :, 1] > 230)
            & (overlay[:, :, 2] < 60)
            & (overlay[:, :, 0] < 60)
        )
        self.assertFalse(result['is_crack'])
        self.assertEqual(result['crack_size'], 'none')
        self.assertEqual(result['egg_size'], 'medium')
        self.assertGreater(result['egg_size_confidence'], 0.8)
        self.assertGreater(result['egg_area_ratio'], 0.2)
        self.assertGreater(result['egg_length_pixels'], result['egg_width_pixels'])
        self.assertIn('shell_texture_score', result)
        self.assertIn('shell_texture_uniformity', result)
        self.assertIn('thin_crack_score', result)
        DetectionResponse.model_validate(result)
        self.assertEqual(red_pixels, 0)
        self.assertGreater(green_pixels, 300)

    def test_horizontal_candled_egg_is_detected(self) -> None:
        result = detect_image_bytes(
            _candled_egg(horizontal=True),
            cfg=self.config,
        )

        self.assertTrue(result['egg_detected'])
        self.assertFalse(result['is_crack'])
        self.assertGreater(result['egg_area_ratio'], 0.2)
        self.assertGreater(
            result['egg_length_pixels'],
            result['egg_width_pixels'],
        )
        DetectionResponse.model_validate(result)

    def test_horizontal_dark_hairline_is_detected(self) -> None:
        result = detect_image_bytes(
            _candled_egg('dark', horizontal=True),
            cfg=self.config,
        )

        self.assertTrue(result['egg_detected'])
        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['candidate_pixels'], 0)
        DetectionResponse.model_validate(result)

    def test_fuzzy_egg_size_memberships(self) -> None:
        self.assertEqual(_fuzzy_egg_size(0.08, self.config)[0], 'small')
        self.assertEqual(_fuzzy_egg_size(0.27, self.config)[0], 'medium')
        self.assertEqual(_fuzzy_egg_size(0.55, self.config)[0], 'large')

    def test_one_pixel_dark_hairline_is_detected_and_traced(self) -> None:
        self._assert_line_crack('dark')

    def test_peripheral_dark_hairline_is_detected(self) -> None:
        result = detect_image_bytes(_peripheral_candled_egg(), cfg=self.config)

        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['candidate_pixels'], 0)
        DetectionResponse.model_validate(result)

    def test_one_pixel_bright_light_leak_is_detected_and_traced(self) -> None:
        self._assert_line_crack('bright')

    def test_one_pixel_low_contrast_hairline_is_detected(self) -> None:
        self._assert_line_crack('subtle')

    def test_fragmented_clean_texture_is_not_a_crack(self) -> None:
        result = detect_image_bytes(
            _candled_egg(textured=True),
            cfg=self.config,
        )
        self.assertGreater(
            result['raw_candidate_components'],
            self.config.max_fragmented_components,
            result,
        )
        self.assertFalse(result['dominant_crack_override'], result)
        self.assertFalse(result['is_crack'], result)

    def test_low_contrast_hairline_survives_shell_texture(self) -> None:
        result = detect_image_bytes(
            _candled_egg('subtle', textured=True),
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['shell_texture_score'], 0.0)

    def test_default_resolution_rejects_texture_and_keeps_hairline(self) -> None:
        clean = detect_image_bytes(_candled_egg(textured=True), cfg=CONFIG)
        cracked = detect_image_bytes(
            _candled_egg('dark', textured=True),
            cfg=CONFIG,
        )
        self.assertFalse(clean['is_crack'], clean)
        self.assertTrue(cracked['is_crack'], cracked)
        self.assertTrue(cracked['dominant_crack_override'], cracked)

    def test_geometry_is_stable_at_double_resolution(self) -> None:
        high_resolution = replace(
            self.config,
            target_width=1920,
            target_height=1080,
        )
        clean = detect_image_bytes(
            _double_resolution(_candled_egg(textured=True)),
            cfg=high_resolution,
        )
        cracked = detect_image_bytes(
            _double_resolution(_candled_egg('dark', textured=True)),
            cfg=high_resolution,
        )
        self.assertFalse(clean['is_crack'], clean)
        self.assertTrue(cracked['is_crack'], cracked)
        self.assertGreater(cracked['egg_width_pixels'], 700.0)

    def test_dominant_hairline_survives_fragmented_texture(self) -> None:
        result = detect_image_bytes(
            _candled_egg('dark', textured=True),
            cfg=self.config,
        )
        self.assertGreater(
            result['raw_candidate_components'],
            self.config.max_fragmented_components,
            result,
        )
        self.assertTrue(result['dominant_crack_override'], result)
        self.assertTrue(result['is_crack'], result)
        self.assertLessEqual(result['candidate_components'], 4)

    def test_smooth_yolk_boundary_is_not_a_crack(self) -> None:
        result = detect_image_bytes(
            _modified_candled_egg('yolk_arc'),
            cfg=self.config,
        )
        self.assertFalse(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 0)

    def test_broad_dark_shell_texture_is_not_a_crack(self) -> None:
        result = detect_image_bytes(
            _modified_candled_egg('dark_texture'),
            cfg=self.config,
        )
        self.assertFalse(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 0)

    def test_clustered_rim_shell_texture_is_not_a_crack(self) -> None:
        result = detect_image_bytes(
            _modified_candled_egg('rim_texture'),
            cfg=self.config,
        )
        self.assertFalse(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 0)

    def test_smooth_shell_fiber_is_not_a_crack(self) -> None:
        result = detect_image_bytes(
            _modified_candled_egg('shell_fiber'),
            cfg=self.config,
        )
        self.assertFalse(result['is_crack'], result)
        self.assertEqual(result['candidate_components'], 0)

    def test_pale_surface_fracture_is_detected(self) -> None:
        result = detect_image_bytes(
            _modified_candled_egg('pale_surface_crack'),
            include_steps=True,
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['candidate_pixels'], 0)
        self.assertGreater(result['contour_length'], 70.0)
        self.assertTrue(result['crack_locations'], result)
        crack_mask = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(result['crack_mask_b64']),
                dtype=np.uint8,
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(crack_mask)
        reference = np.zeros_like(crack_mask)
        cv2.polylines(reference, [np.array([
            [292, 352], [323, 335], [358, 333],
        ], dtype=np.int32)], False, 255, 1, cv2.LINE_8)
        cv2.polylines(reference, [np.array([
            [365, 330], [398, 309],
        ], dtype=np.int32)], False, 255, 1, cv2.LINE_8)
        tolerance = cv2.dilate(
            crack_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        recall = cv2.countNonZero(cv2.bitwise_and(reference, tolerance)) \
            / max(float(cv2.countNonZero(reference)), 1.0)
        self.assertGreaterEqual(recall, 0.70, result)
        steps = result['intermediate_steps']
        self.assertIn('perimeter_shell_zone', steps)
        self.assertIn('pale_surface_response', steps)
        self.assertIn('fused_crack_response', steps)

    def test_broad_flashlight_glare_requests_recapture(self) -> None:
        with self.assertRaisesRegex(DetectionError, 'glare'):
            detect_image_bytes(
                _modified_candled_egg('glare'),
                cfg=self.config,
            )

    def test_blurred_crack_requests_recapture(self) -> None:
        with self.assertRaisesRegex(DetectionError, 'blurry'):
            detect_image_bytes(
                _modified_candled_egg('blur'),
                cfg=self.config,
            )

    def test_native_image_is_not_upscaled(self) -> None:
        result = detect_image_bytes(_candled_egg('dark'), cfg=CONFIG)
        original = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(result['original_image_b64']),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(original)
        self.assertEqual(original.shape[:2], (540, 960))
        self.assertGreater(result['image_quality_score'], 0.0)
        self.assertFalse(result['requires_recapture'])

    def test_sparse_strong_crack_network_can_override_fragmentation(self) -> None:
        metrics = {
            'skeleton_length': 961.0,
            'span': 488.8,
            'elongation': 2.88,
            'average_thickness': 7.9,
            'density': 0.081,
            'strength_p90': 141.0,
            'strong_overlap': 0.61,
            'score': 5.35,
        }
        component = CrackComponent(
            0,
            0,
            np.zeros((1, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            metrics,
            'combined',
        )
        self.assertTrue(_is_dominant_crack_component(component, self.config))

    def test_aligned_hairline_fragments_form_a_dominant_group(self) -> None:
        components = []
        for x in (0, 100, 200):
            metrics = {
                'skeleton_length': 120.0,
                'span': 40.0,
                'elongation': 6.0,
                'average_thickness': 3.0,
                'density': 0.12,
                'strength_p90': 80.0,
                'axis_x': 1.0,
                'axis_y': 0.0,
                'center_x': 20.0,
                'center_y': 2.0,
                'endpoint_a_x': 0.0,
                'endpoint_a_y': 2.0,
                'endpoint_b_x': 40.0,
                'endpoint_b_y': 2.0,
            }
            components.append(CrackComponent(
                x,
                0,
                np.zeros((5, 40), dtype=np.uint8),
                np.zeros((5, 40), dtype=np.uint8),
                metrics,
                'combined',
            ))
        group = _dominant_fragment_group(components, self.config)
        self.assertEqual(len(group), 3)

    def _assert_line_crack(self, crack: str) -> None:
        result = detect_image_bytes(_candled_egg(crack), cfg=self.config)
        overlay = _decode_overlay(result)
        red_pixels = int(np.count_nonzero(
            (overlay[:, :, 2] > 230)
            & (overlay[:, :, 1] < 60)
            & (overlay[:, :, 0] < 60)
        ))
        self.assertTrue(result['is_crack'], result)
        self.assertIn(result['crack_size'], {'small', 'medium', 'large'})
        self.assertGreater(result['contour_length'], 35)
        self.assertGreater(red_pixels, 50)
        self.assertLess(red_pixels, result['contour_length'] * 12)

    def test_faint_dark_crack_is_detected(self) -> None:
        """A 1-pixel dark crack only ~12 grey levels below the shell."""
        result = detect_image_bytes(
            _candled_egg('faint_dark'),
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['candidate_pixels'], 0)

    def test_very_faint_dark_crack_on_textured_shell(self) -> None:
        """A dark crack (30 grey levels) must survive shell texture noise."""
        result = detect_image_bytes(
            _candled_egg('dark_on_texture', textured=True),
            cfg=self.config,
        )
        self.assertTrue(result['is_crack'], result)
        self.assertGreater(result['candidate_pixels'], 0)


if __name__ == '__main__':
    unittest.main()
