import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/di/service_locator.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../services/camera_session_service.dart';
import '../../../../services/error_message_service.dart';
import '../../../../shared/widgets/crack_divider.dart';
import '../../../../shared/widgets/scan_line_overlay.dart';
import '../../../../shared/widgets/viewfinder_frame.dart';
import '../../../detection/domain/detection_result.dart';
import '../../../detection/presentation/widgets/verdict_badge.dart';
import '../../domain/camera_repository.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final CameraRepository _repository = getIt<CameraRepository>();
  final CameraSessionService _cameraSession = getIt<CameraSessionService>();

  DetectionResult? _result;
  Uint8List? _resultOverlayBytes;
  String? _errorMessage;
  String? _captureName;
  bool _capturing = false;
  bool _saving = false;
  bool _saved = false;
  bool _cameraWasReady = false;
  bool _autoFocusScheduled = false;

  @override
  void initState() {
    super.initState();
    _cameraSession.addListener(_onCameraChanged);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_startCamera());
    });
  }

  Future<void> _startCamera() async {
    await _cameraSession.start();
    if (mounted) _onCameraChanged();
  }

  @override
  void dispose() {
    _cameraSession.removeListener(_onCameraChanged);
    super.dispose();
  }

  void _onCameraChanged() {
    final ready = _cameraSession.isReady && !_cameraSession.initializing;
    if (!ready) {
      _cameraWasReady = false;
      _autoFocusScheduled = false;
    } else if (!_cameraWasReady &&
        !_autoFocusScheduled &&
        !_cameraSession.focusLocked) {
      _autoFocusScheduled = true;
      unawaited(_autoFocusAfterCameraLoad());
    }
    _cameraWasReady = ready;
    if (mounted) setState(() {});
  }

  Future<void> _autoFocusAfterCameraLoad() async {
    await Future<void>.delayed(const Duration(milliseconds: 800));
    if (!mounted ||
        !_cameraSession.isReady ||
        _cameraSession.initializing ||
        _cameraSession.focusing ||
        _cameraSession.focusLocked) {
      return;
    }
    await _calibrateIlluminatedEggFocus(showErrors: false);
  }

  Future<void> _focusOnIlluminatedEgg() async {
    await _calibrateIlluminatedEggFocus();
  }

  Future<bool> _calibrateIlluminatedEggFocus({bool showErrors = true}) async {
    if (_cameraSession.focusing || !_cameraSession.isReady || _capturing) {
      return false;
    }

    if (showErrors) setState(() => _errorMessage = null);
    try {
      final focused = await _cameraSession.focusOnIlluminatedEgg((frame) async {
        final bytes = await frame.readAsBytes();
        return _repository.scoreFocusFrame(
          imageBytes: bytes,
          filename: frame.name.isEmpty ? 'focus-frame.jpg' : frame.name,
        );
      });

      if (!focused && mounted && showErrors) {
        setState(() {
          _errorMessage =
              'The webcam does not support manual focus calibration.';
        });
      }
      return focused;
    } catch (error) {
      if (mounted && showErrors) {
        setState(() {
          _errorMessage = friendlyErrorMessage(
            error,
            fallback: 'Could not focus on the illuminated egg.',
          );
        });
      }
      return false;
    }
  }

  Future<void> _captureAndDetect() async {
    if (_capturing || !_cameraSession.isReady) return;

    if (!_cameraSession.focusLocked) {
      final focused = await _calibrateIlluminatedEggFocus();
      if (!focused || !mounted) return;
    }

    setState(() {
      _capturing = true;
      _saving = false;
      _saved = false;
      _errorMessage = null;
      _result = null;
      _resultOverlayBytes = null;
    });

    try {
      final picture = await _cameraSession.takePicture();
      final bytes = await picture.readAsBytes();
      if (bytes.isEmpty) {
        throw StateError('The captured image is empty.');
      }

      final filename = picture.name.isNotEmpty
          ? picture.name
          : 'camera-${DateTime.now().millisecondsSinceEpoch}.jpg';

      final result = await _repository.detectCapturedImage(
        imageBytes: bytes,
        filename: filename,
      );

      if (!mounted) return;
      setState(() {
        _capturing = false;
        _captureName = filename;
        _result = result;
        _resultOverlayBytes = _decodeImage(result.overlayImageB64);
      });
      unawaited(_cameraSession.ensurePreview());
    } on CameraException catch (error) {
      if (!mounted) return;
      setState(() {
        _capturing = false;
        _errorMessage = cameraErrorMessage(error);
      });
      unawaited(_cameraSession.ensurePreview());
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _capturing = false;
        _errorMessage = friendlyErrorMessage(
          error,
          fallback: 'The image could not be captured or analyzed.',
        );
      });
      unawaited(_cameraSession.ensurePreview());
    }
  }

  Future<void> _saveResult() async {
    final result = _result;
    if (result == null || _saving || _saved) return;

    setState(() {
      _saving = true;
      _errorMessage = null;
    });

    try {
      await _repository.saveDetection(
        result: result,
        sourceName: _captureName ?? 'camera',
      );
      if (!mounted) return;
      setState(() {
        _saving = false;
        _saved = true;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _saving = false;
        _errorMessage = friendlyErrorMessage(
          error,
          fallback: 'The detection could not be saved.',
        );
      });
    }
  }

  Future<void> _backToPreview() async {
    setState(() {
      _result = null;
      _resultOverlayBytes = null;
      _captureName = null;
      _saved = false;
      _saving = false;
      _errorMessage = null;
    });
    await _cameraSession.ensurePreview();
  }

  Uint8List? _decodeImage(String value) {
    if (value.isEmpty) return null;
    try {
      return base64Decode(value);
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final viewfinder = _buildViewfinder();
    final controls = _buildControls();

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Responsive.spaceLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('LIVE CAMERA', style: AppTheme.display(22)),
              const SizedBox(height: 2),
              const Text(
                'Native Flutter camera preview with Python image analysis.',
                style: TextStyle(color: AppColors.shellMuted),
              ),
              const SizedBox(height: 12),
              const CrackDivider(),
              const SizedBox(height: Responsive.spaceLg),
              Expanded(
                child: Responsive.isMobile(context)
                    ? SingleChildScrollView(
                        child: Column(
                          children: [
                            viewfinder,
                            const SizedBox(height: Responsive.spaceLg),
                            controls,
                          ],
                        ),
                      )
                    : Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(flex: 2, child: viewfinder),
                          const SizedBox(width: Responsive.spaceLg),
                          SizedBox(width: 340, child: controls),
                        ],
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildViewfinder() {
    final controller = _cameraSession.controller;
    double aspectRatio = 16 / 9;
    if (controller != null && controller.value.isInitialized) {
      final size = controller.value.previewSize;
      if (size != null && size.width > 0 && size.height > 0) {
        aspectRatio = size.width / size.height;
      }
    }

    return Container(
      constraints: const BoxConstraints(minHeight: 420),
      decoration: BoxDecoration(
        color: Colors.black,
        border: Border.all(color: AppColors.hairline),
        borderRadius: BorderRadius.circular(4),
      ),
      clipBehavior: Clip.antiAlias,
      child: AspectRatio(
        aspectRatio: aspectRatio,
        child: ViewfinderFrame(
          child: Stack(
            fit: StackFit.expand,
            children: [
              _buildPreviewContent(),
              ScanLineOverlay(active: _capturing),
              Positioned(
                left: 16,
                top: 16,
                child: _StatusPill(
                  active: _cameraSession.isReady,
                  text: _result == null
                      ? _cameraSession.statusMessage
                      : 'Detection complete',
                ),
              ),
              if (_result != null)
                Positioned(
                  right: 16,
                  top: 16,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.surface.withValues(alpha: 0.92),
                      border: Border.all(color: AppColors.hairline),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: VerdictBadge(isCrack: _result!.isCrack),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPreviewContent() {
    if (_resultOverlayBytes != null) {
      return Image.memory(
        _resultOverlayBytes!,
        fit: BoxFit.contain,
        gaplessPlayback: true,
      );
    }

    final controller = _cameraSession.controller;
    if (controller != null && controller.value.isInitialized) {
      return _buildLandscapePreview(controller);
    }

    if (_cameraSession.detecting || _cameraSession.initializing) {
      return const Center(child: CircularProgressIndicator());
    }

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.videocam_off_outlined,
              color: AppColors.shellFaint,
              size: 52,
            ),
            const SizedBox(height: 14),
            Text(
              _cameraSession.errorMessage ?? 'Waiting for a camera...',
              textAlign: TextAlign.center,
              style: const TextStyle(color: AppColors.shellMuted),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLandscapePreview(CameraController controller) {
    final size = controller.value.previewSize;
    if (size == null || size.width <= 0 || size.height <= 0) {
      return controller.buildPreview();
    }

    return ClipRect(
      child: FittedBox(
        fit: BoxFit.contain,
        alignment: Alignment.center,
        child: SizedBox(
          width: size.width,
          height: size.height,
          child: controller.buildPreview(),
        ),
      ),
    );
  }

  Widget _buildControls() {
    final cameras = _cameraSession.cameras;
    final selected = _cameraSession.selectedCameraIndex;
    final result = _result;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border.all(color: AppColors.hairline),
        borderRadius: BorderRadius.circular(4),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('CAMERA CONTROL', style: AppTheme.display(16)),
            const SizedBox(height: 18),
            DropdownButtonFormField<int>(
              key: ValueKey('${cameras.length}-$selected'),
              initialValue: selected,
              isExpanded: true,
              decoration: const InputDecoration(labelText: 'Detected camera'),
              selectedItemBuilder: (context) => [
                for (var i = 0; i < cameras.length; i++)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Camera ${i + 1}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
              ],
              items: [
                for (var i = 0; i < cameras.length; i++)
                  DropdownMenuItem<int>(
                    value: i,
                    child: Tooltip(
                      message: cameras[i].name,
                      child: Text(
                        _cameraSession.cameraLabel(cameras[i], i),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
              ],
              onChanged: _capturing ||
                      _cameraSession.detecting ||
                      _cameraSession.initializing
                  ? null
                  : (index) {
                      if (index != null) {
                        unawaited(_cameraSession.selectCamera(index));
                      }
                    },
            ),
            const SizedBox(height: 14),
            _ReadoutRow(label: 'DETECTED', value: '${cameras.length}'),
            _ReadoutRow(
              label: 'STATUS',
              value: _cameraSession.statusMessage,
            ),
            _ReadoutRow(
              label: 'ORIENTATION',
              value: 'Landscape',
            ),
            _ReadoutRow(
              label: 'AUTOFOCUS',
              value: _cameraSession.autofocusStatus,
            ),
            _ReadoutRow(
              label: 'QUALITY',
              value: _cameraSession.resolutionStatus,
            ),
            if (_cameraSession.errorMessage != null ||
                _errorMessage != null) ...[
              const SizedBox(height: 14),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.rustDim,
                  border: Border.all(color: AppColors.rust),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  _errorMessage ?? _cameraSession.errorMessage!,
                  style: const TextStyle(color: AppColors.crackBadgeText),
                ),
              ),
            ],
            if (result != null) ...[
              const SizedBox(height: 16),
              const Divider(),
              const SizedBox(height: 12),
              VerdictBadge(isCrack: result.isCrack),
              const SizedBox(height: 12),
              _ReadoutRow(
                label: 'CONFIDENCE',
                value: '${(result.confidence * 100).toStringAsFixed(1)}%',
              ),
              _ReadoutRow(
                label: 'COMPONENTS',
                value: '${result.candidateComponents}',
              ),
              _ReadoutRow(
                label: 'SCORE',
                value: result.detectionScore.toStringAsFixed(2),
              ),
              _ReadoutRow(
                label: 'EGG SIZE',
                value:
                    '${result.eggSize.toUpperCase()} ${(result.eggSizeConfidence * 100).toStringAsFixed(1)}%',
              ),
              _ReadoutRow(
                label: 'CRACK SIZE',
                value: result.crackSize.toUpperCase(),
              ),
              _ReadoutRow(
                label: 'CRACK DETAIL',
                value: result.thinCrackDetected ? 'THIN HAIRLINE' : 'STANDARD',
              ),
              _ReadoutRow(
                label: 'PROCESSING',
                value: '${result.processingTimeMs} ms',
              ),
              if (_saved)
                const Padding(
                  padding: EdgeInsets.only(top: 10),
                  child: Text(
                    'Saved to history',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: AppColors.sage),
                  ),
                ),
            ],
            const SizedBox(height: 20),
            if (result == null) ...[
              OutlinedButton.icon(
                onPressed: _cameraSession.isReady &&
                        !_capturing &&
                        !_cameraSession.focusing
                    ? () => unawaited(_focusOnIlluminatedEgg())
                    : null,
                icon: const Icon(Icons.center_focus_strong_outlined),
                label: Text(
                  _cameraSession.focusing
                      ? 'Scanning egg focus...'
                      : 'Focus on lit egg',
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton.icon(
                onPressed: _cameraSession.isReady &&
                        !_capturing &&
                        !_cameraSession.focusing
                    ? _captureAndDetect
                    : null,
                icon: const Icon(Icons.camera_alt_outlined),
                label:
                    Text(_capturing ? 'Processing...' : 'Capture and detect'),
              )
            ] else ...[
              ElevatedButton.icon(
                onPressed: _saving || _saved ? null : _saveResult,
                icon: Icon(_saved ? Icons.check : Icons.save_outlined),
                label: Text(
                  _saved
                      ? 'Saved'
                      : _saving
                          ? 'Saving...'
                          : 'Save to history',
                ),
              ),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: _saving ? null : _backToPreview,
                icon: const Icon(Icons.arrow_back),
                label: const Text('Back to camera preview'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  final bool active;
  final String text;

  const _StatusPill({required this.active, required this.text});

  @override
  Widget build(BuildContext context) {
    final color = active ? AppColors.sage : AppColors.amber;

    return Container(
      constraints: const BoxConstraints(maxWidth: 360),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.surface.withValues(alpha: 0.92),
        border: Border.all(color: color),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(shape: BoxShape.circle, color: color),
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              text,
              overflow: TextOverflow.ellipsis,
              style: AppTheme.mono(11, color: color),
            ),
          ),
        ],
      ),
    );
  }
}

class _ReadoutRow extends StatelessWidget {
  final String label;
  final String value;

  const _ReadoutRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 92,
            child: Text(
              label,
              style: AppTheme.mono(
                10,
                color: AppColors.shellFaint,
                weight: FontWeight.w600,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.right,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: AppTheme.mono(11, color: AppColors.shellMuted),
            ),
          ),
        ],
      ),
    );
  }
}
