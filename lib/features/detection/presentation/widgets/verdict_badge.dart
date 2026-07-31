import 'package:flutter/material.dart';
import '../../../../core/constants/app_colors.dart';
import '../../../../core/theme/app_theme.dart';

/// Status indicator styled like an instrument-panel LED rather than a
/// rounded chip -- a small glowing dot plus a mono-set label, matching
/// the console's data-readout language.
class VerdictBadge extends StatelessWidget {
  final bool isCrack;

  const VerdictBadge({super.key, required this.isCrack});

  @override
  Widget build(BuildContext context) {
    final color = isCrack ? AppColors.rust : AppColors.sage;
    final label = isCrack ? 'CRACK DETECTED' : 'NO CRACK';

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color,
            boxShadow: [BoxShadow(color: color.withValues(alpha: 0.7), blurRadius: 8, spreadRadius: 1)],
          ),
        ),
        const SizedBox(width: 10),
        Text(label, style: AppTheme.mono(13, weight: FontWeight.w600, color: color)),
      ],
    );
  }
}
