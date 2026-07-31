import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';

/// The app's signature motif: a thin, deliberately irregular line — the
/// same visual language as the crack overlay drawn on inspected eggs —
/// reused as a structural divider instead of a plain hairline rule.
///
/// The jaggedness is seeded from [seed] so it's stable across rebuilds
/// but varies between instances, avoiding a repeated decorative pattern.
class CrackDivider extends StatelessWidget {
  final double height;
  final Color color;
  final int seed;

  const CrackDivider({
    super.key,
    this.height = 10,
    this.color = AppColors.hairline,
    this.seed = 7,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      width: double.infinity,
      child: CustomPaint(
        painter: _CrackLinePainter(color: color, seed: seed),
      ),
    );
  }
}

class _CrackLinePainter extends CustomPainter {
  final Color color;
  final int seed;

  _CrackLinePainter({required this.color, required this.seed});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke;

    final rnd = _LcgRandom(seed);
    final path = Path();
    final midY = size.height / 2;
    path.moveTo(0, midY);

    const segmentWidth = 18.0;
    var x = 0.0;
    while (x < size.width) {
      final nextX = (x + segmentWidth).clamp(0, size.width).toDouble();
      final jitter = (rnd.nextDouble() - 0.5) * size.height * 0.9;
      path.lineTo(nextX, midY + jitter);
      x = nextX;
    }
    path.lineTo(size.width, midY);

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _CrackLinePainter oldDelegate) =>
      oldDelegate.color != color || oldDelegate.seed != seed;
}

/// Minimal deterministic PRNG so the "crack" shape is stable per seed
/// without pulling in dart:math's Random(seed) platform differences.
class _LcgRandom {
  int _state;
  _LcgRandom(int seed) : _state = seed == 0 ? 1 : seed;

  double nextDouble() {
    _state = (_state * 1103515245 + 12345) & 0x7fffffff;
    return _state / 0x7fffffff;
  }
}
