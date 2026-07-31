import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';

/// Corner-bracket frame around an image or live preview, evoking the
/// paper's dark-box + flashlight rig where eggs are actually photographed.
/// Reused on both the detection and camera screens so the two feel like
/// one instrument rather than two unrelated pages.
class ViewfinderFrame extends StatelessWidget {
  final Widget child;
  final Color bracketColor;
  final double bracketLength;
  final double bracketThickness;

  const ViewfinderFrame({
    super.key,
    required this.child,
    this.bracketColor = AppColors.amber,
    this.bracketLength = 22,
    this.bracketThickness = 2,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(child: child),
        ..._corners(),
      ],
    );
  }

  List<Widget> _corners() {
    Widget bracket({required bool top, required bool left}) {
      return Positioned(
        top: top ? 0 : null,
        bottom: top ? null : 0,
        left: left ? 0 : null,
        right: left ? null : 0,
        child: CustomPaint(
          size: Size(bracketLength, bracketLength),
          painter: _CornerPainter(
            color: bracketColor,
            thickness: bracketThickness,
            top: top,
            left: left,
          ),
        ),
      );
    }

    return [
      bracket(top: true, left: true),
      bracket(top: true, left: false),
      bracket(top: false, left: true),
      bracket(top: false, left: false),
    ];
  }
}

class _CornerPainter extends CustomPainter {
  final Color color;
  final double thickness;
  final bool top;
  final bool left;

  _CornerPainter({
    required this.color,
    required this.thickness,
    required this.top,
    required this.left,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = thickness
      ..strokeCap = StrokeCap.square
      ..style = PaintingStyle.stroke;

    final path = Path();
    final y = top ? 0.0 : size.height;
    final x = left ? 0.0 : size.width;
    final yInset = top ? size.height : 0.0;
    final xInset = left ? size.width : 0.0;

    path.moveTo(x, yInset);
    path.lineTo(x, y);
    path.lineTo(xInset, y);

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _CornerPainter oldDelegate) => false;
}
