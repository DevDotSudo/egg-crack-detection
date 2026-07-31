import 'dart:convert';
import 'package:flutter/material.dart';
import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';

class ImageCompareView extends StatelessWidget {
  final String originalB64;
  final String overlayB64;

  const ImageCompareView({
    super.key,
    required this.originalB64,
    required this.overlayB64,
  });

  @override
  Widget build(BuildContext context) {
    final isNarrow = Responsive.isMobile(context);
    final children = [
      Expanded(child: _LabeledImage(label: 'Original', b64: originalB64)),
      SizedBox(
        width: isNarrow ? 0 : Responsive.spaceMd,
        height: isNarrow ? Responsive.spaceMd : 0,
      ),
      Expanded(child: _LabeledImage(label: 'Overlay (crack line)', b64: overlayB64)),
    ];

    return isNarrow ? Column(children: children) : Row(children: children);
  }
}

class _LabeledImage extends StatelessWidget {
  final String label;
  final String b64;

  const _LabeledImage({required this.label, required this.b64});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: Theme.of(context).textTheme.labelLarge),
        const SizedBox(height: 8),
        AspectRatio(
          aspectRatio: 1147 / 633, // matches the paper's rescaled dimensions
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: AppColors.border),
              borderRadius: BorderRadius.circular(8),
            ),
            clipBehavior: Clip.antiAlias,
            child: b64.isEmpty
                ? const Center(child: Icon(Icons.image_outlined, size: 40))
                : Image.memory(base64Decode(b64), fit: BoxFit.contain),
          ),
        ),
      ],
    );
  }
}
