import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.core.config import CONFIG
from app.detection.fuzzy import CrackSizeMamdaniClassifier, EggSizeMamdaniClassifier, MamdaniEngine
from app.services.detector import _fuzzy_apparent_egg_size, _fuzzy_crack_size, _fuzzy_egg_size


class MamdaniTests(unittest.TestCase):
    def test_engine_uses_centroid_defuzzification(self) -> None:
        result = MamdaniEngine(CONFIG.fuzzy).infer([(1.0, 'medium')])
        self.assertEqual(result.label, 'medium')
        self.assertAlmostEqual(result.score, 0.5, places=2)
        self.assertGreater(result.confidence, 0.70)

    def test_egg_size_rules_classify_three_sizes(self) -> None:
        self.assertEqual(_fuzzy_egg_size(38.0, 51.0, CONFIG)[0], 'small')
        self.assertEqual(_fuzzy_egg_size(43.0, 57.0, CONFIG)[0], 'medium')
        self.assertEqual(_fuzzy_egg_size(49.0, 65.0, CONFIG)[0], 'large')

    def test_egg_size_uses_only_width_and_height_mm(self) -> None:
        result = EggSizeMamdaniClassifier(CONFIG.fuzzy).classify(43.0, 57.0)
        self.assertEqual(result.label, 'medium')
        self.assertAlmostEqual(sum(result.memberships.values()), 1.0, places=6)

    def test_mixed_dimensions_use_mamdani_rules(self) -> None:
        small = EggSizeMamdaniClassifier(CONFIG.fuzzy).classify(39.0, 56.0)
        large = EggSizeMamdaniClassifier(CONFIG.fuzzy).classify(46.0, 60.0)
        self.assertEqual(small.label, 'small')
        self.assertEqual(large.label, 'large')

    def test_apparent_4_inch_height_width_rules_classify_three_sizes(self) -> None:
        small = _fuzzy_apparent_egg_size(150.0, 210.0, (720, 1280), CONFIG)
        medium = _fuzzy_apparent_egg_size(385.0, 500.0, (720, 1280), CONFIG)
        large = _fuzzy_apparent_egg_size(620.0, 700.0, (720, 1280), CONFIG)
        self.assertEqual(small[0], 'small')
        self.assertEqual(medium[0], 'medium')
        self.assertEqual(large[0], 'large')

    def test_apparent_size_is_orientation_invariant(self) -> None:
        landscape = _fuzzy_apparent_egg_size(385.0, 500.0, (720, 1280), CONFIG)
        portrait = _fuzzy_apparent_egg_size(385.0, 500.0, (1280, 720), CONFIG)
        self.assertEqual(landscape[0], portrait[0])
        self.assertAlmostEqual(landscape[3], portrait[3], places=6)

    def test_crack_size_rules_classify_three_sizes(self) -> None:
        small = _fuzzy_crack_size(True, 40, 15, 100000, 1, 30, CONFIG)
        medium = _fuzzy_crack_size(True, 250, 500, 100000, 4, 65, CONFIG)
        large = _fuzzy_crack_size(True, 600, 4000, 100000, 8, 100, CONFIG)
        self.assertEqual(small[0], 'small')
        self.assertEqual(medium[0], 'medium')
        self.assertEqual(large[0], 'large')

    def test_no_crack_returns_none(self) -> None:
        result = CrackSizeMamdaniClassifier(CONFIG.fuzzy).classify(False, 0, 0, 100000, 0, 0)
        self.assertEqual(result.label, 'none')
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.score, 0.0)


if __name__ == '__main__':
    unittest.main()
