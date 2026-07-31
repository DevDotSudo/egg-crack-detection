import 'dart:convert';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/di/service_locator.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../shared/widgets/crack_divider.dart';
import '../../../../shared/widgets/scan_line_overlay.dart';
import '../../../../shared/widgets/viewfinder_frame.dart';
import '../cubit/detection_cubit.dart';
import '../cubit/detection_state.dart';
import '../widgets/verdict_badge.dart';

class DetectionScreen extends StatelessWidget {
  const DetectionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => getIt<DetectionCubit>(),
      child: const _DetectionView(),
    );
  }
}

class _DetectionView extends StatelessWidget {
  const _DetectionView();

  Future<void> _pickAndRun(BuildContext context) async {
    final cubit = context.read<DetectionCubit>();
    const typeGroup = XTypeGroup(
      label: 'images',
      extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    );
    final file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file == null) return;

    cubit.imagePicked(file.name);
    final bytes = await file.readAsBytes();
    if (!context.mounted) return;
    await cubit.runDetection(bytes, file.name);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Responsive.spaceLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('DETECT FROM IMAGE', style: AppTheme.display(22)),
              const SizedBox(height: 2),
              Text(
                'Red/green channel processing, morphology, contour analysis, and fuzzy grading.',
                style: TextStyle(color: AppColors.shellMuted),
              ),
              const SizedBox(height: 12),
              const CrackDivider(seed: 3),
              const SizedBox(height: Responsive.spaceLg),
              Expanded(
                child: BlocBuilder<DetectionCubit, DetectionState>(
                  builder: (context, state) {
                    final viewfinder = _ViewfinderPanel(state: state);
                    final controls = _ControlPanel(
                      state: state,
                      onPick: () => _pickAndRun(context),
                    );

                    if (Responsive.isMobile(context)) {
                      return SingleChildScrollView(
                        child: Column(
                          children: [
                            viewfinder,
                            const SizedBox(height: Responsive.spaceLg),
                            controls,
                          ],
                        ),
                      );
                    }

                    return Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(flex: 2, child: viewfinder),
                        const SizedBox(width: Responsive.spaceLg),
                        SizedBox(width: 340, child: controls),
                      ],
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ViewfinderPanel extends StatelessWidget {
  final DetectionState state;

  const _ViewfinderPanel({required this.state});

  @override
  Widget build(BuildContext context) {
    final loading = state.status == DetectionStatus.loading;
    final hasResult =
        state.status == DetectionStatus.success && state.result != null;

    return AspectRatio(
      aspectRatio: 1147 / 633,
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          border: Border.all(color: AppColors.hairline),
        ),
        child: ViewfinderFrame(
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (hasResult)
                Padding(
                  padding: const EdgeInsets.all(8),
                  child: Image.memory(
                    base64Decode(state.result!.overlayImageB64),
                    fit: BoxFit.contain,
                  ),
                )
              else
                Container(
                  color: AppColors.ink,
                  alignment: Alignment.center,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        loading
                            ? Icons.hourglass_empty
                            : Icons.image_search_outlined,
                        color: AppColors.shellFaint,
                        size: 40,
                      ),
                      const SizedBox(height: 10),
                      Text(
                        loading ? 'PROCESSING...' : 'CHOOSE AN IMAGE TO BEGIN',
                        style: AppTheme.mono(12, color: AppColors.shellFaint),
                      ),
                    ],
                  ),
                ),
              ScanLineOverlay(active: loading),
              if (hasResult)
                Positioned(
                  left: 16,
                  bottom: 16,
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    color: AppColors.ink.withValues(alpha: 0.75),
                    child: VerdictBadge(isCrack: state.result!.isCrack),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ControlPanel extends StatelessWidget {
  final DetectionState state;
  final VoidCallback onPick;

  const _ControlPanel({required this.state, required this.onPick});

  @override
  Widget build(BuildContext context) {
    final cubit = context.read<DetectionCubit>();

    return Container(
      padding: const EdgeInsets.all(Responsive.spaceMd),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border.all(color: AppColors.hairline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'SOURCE FILE',
            style: AppTheme.mono(11, color: AppColors.shellFaint),
          ),
          const SizedBox(height: 6),
          Text(
            state.selectedFilename ?? 'none selected',
            style: AppTheme.mono(12, color: AppColors.shell),
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'PIPELINE STEPS VIEW',
                style: AppTheme.mono(11, color: AppColors.shellFaint),
              ),
              Switch(
                value: state.showPipelineSteps,
                onChanged: (value) => cubit.togglePipelineSteps(value),
              ),
            ],
          ),
          const SizedBox(height: 16),
          const CrackDivider(height: 6, seed: 3),
          const SizedBox(height: 16),
          if (state.status == DetectionStatus.success && state.result != null)
            ...[
              _ReadoutRow(
                label: 'CONFIDENCE',
                value:
                    '${(state.result!.confidence * 100).toStringAsFixed(1)}%',
              ),
              _ReadoutRow(
                label: 'CRACK SIZE',
                value: _labelWithConfidence(
                  state.result!.crackSize,
                  state.result!.crackSizeConfidence,
                ),
              ),
              _ReadoutRow(
                label: 'EGG SIZE',
                value: _labelWithConfidence(
                  state.result!.eggSize,
                  state.result!.eggSizeConfidence,
                ),
              ),
              _ReadoutRow(
                label: 'CRACK AREA RATIO',
                value: state.result!.areaRatio.toStringAsFixed(6),
              ),
              _ReadoutRow(
                label: 'EGG AREA RATIO',
                value: state.result!.eggAreaRatio.toStringAsFixed(6),
              ),
              _ReadoutRow(
                label: 'CONTOUR LENGTH',
                value: state.result!.contourLength.toStringAsFixed(1),
              ),
              _ReadoutRow(
                label: 'CRACK DETAIL',
                value: state.result!.thinCrackDetected
                    ? 'THIN HAIRLINE'
                    : 'STANDARD',
              ),
              _ReadoutRow(
                label: 'PROCESSING TIME',
                value: '${state.result!.processingTimeMs} ms',
              ),
            ]
          else if (state.status == DetectionStatus.failure)
            Text(
              state.errorMessage ?? 'Something went wrong.',
              style: TextStyle(color: AppColors.rust, fontSize: 12),
            )
          else
            Text(
              'Readouts appear here after detection runs.',
              style: TextStyle(color: AppColors.shellFaint, fontSize: 12),
            ),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: state.status == DetectionStatus.loading ? null : onPick,
              icon: const Icon(Icons.upload_file_outlined),
              label: Text(
                state.selectedFilename == null
                    ? 'Choose image'
                    : 'Choose different image',
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _labelWithConfidence(String label, double confidence) {
    if (label.isEmpty) return '-';
    return '${label.toUpperCase()} ${(confidence * 100).toStringAsFixed(1)}%';
  }
}

class _ReadoutRow extends StatelessWidget {
  final String label;
  final String value;

  const _ReadoutRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: AppTheme.mono(11, color: AppColors.shellFaint)),
          Flexible(
            child: Text(
              value,
              style: AppTheme.mono(12, color: AppColors.shell),
              textAlign: TextAlign.right,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
