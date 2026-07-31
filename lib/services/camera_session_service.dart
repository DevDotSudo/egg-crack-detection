import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'error_message_service.dart';
import 'windows_camera_focus_service.dart';

typedef FocusFrameScorer = Future<double> Function(XFile frame);

class CameraSessionService extends ChangeNotifier with WidgetsBindingObserver {
  List<CameraDescription> _cameras = const [];
  CameraController? _controller;
  int? _selectedCameraIndex;
  Future<void>? _operation;
  Future<bool>? _focusOperation;
  Timer? _retryTimer;
  bool _detecting = false;
  bool _initializing = false;
  bool _shutdown = false;
  bool _suspended = false;
  bool _focusLocked = false;
  String _statusMessage = 'Detecting cameras...';
  String? _errorMessage;
  String _autofocusStatus = 'Camera driver';
  String _resolutionStatus = 'Maximum available';

  CameraSessionService() {
    WidgetsBinding.instance.addObserver(this);
  }

  List<CameraDescription> get cameras => List.unmodifiable(_cameras);
  CameraController? get controller => _controller;
  int? get selectedCameraIndex => _selectedCameraIndex;
  bool get detecting => _detecting;
  bool get initializing => _initializing;
  String get statusMessage => _statusMessage;
  String? get errorMessage => _errorMessage;
  String get autofocusStatus => _autofocusStatus;
  String get resolutionStatus => _resolutionStatus;
  bool get focusing => _focusOperation != null;
  bool get focusLocked => _focusLocked;

  bool get isReady {
    final current = _controller;
    return current != null &&
        current.value.isInitialized &&
        !current.value.hasError;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    switch (state) {
      case AppLifecycleState.resumed:
        _suspended = false;
        unawaited(start(forceRescan: _cameras.isEmpty));
        break;
      case AppLifecycleState.inactive:
        // A Windows desktop app becomes inactive whenever focus moves to a
        // different window. Releasing here made the USB camera close and
        // reopen every time the operator changed windows or controls.
        if (Platform.isAndroid || Platform.isIOS) {
          _suspended = true;
          unawaited(_releaseForPause());
        }
        break;
      case AppLifecycleState.paused:
      case AppLifecycleState.hidden:
        _suspended = true;
        unawaited(_releaseForPause());
        break;
      case AppLifecycleState.detached:
        unawaited(shutdown());
        break;
    }
  }

  Future<void> start({bool forceRescan = false}) async {
    if (_shutdown || _suspended) return;
    final running = _operation;
    if (running != null) {
      await running;
      if (!_shutdown && !_suspended && !isReady) {
        await start(forceRescan: _cameras.isEmpty);
      }
      return;
    }

    late final Future<void> operation;
    operation = _startInternal(forceRescan: forceRescan);
    _operation = operation;
    try {
      await operation;
    } finally {
      if (identical(_operation, operation)) _operation = null;
    }
  }

  Future<void> _startInternal({required bool forceRescan}) async {
    if (isReady && !forceRescan) {
      await ensurePreview();
      return;
    }

    if (forceRescan || _cameras.isEmpty) {
      _detecting = true;
      _initializing = false;
      _statusMessage = 'Detecting cameras...';
      _errorMessage = null;
      _notify();

      try {
        _cameras = await availableCameras().timeout(
          const Duration(seconds: 15),
        );
      } on TimeoutException {
        _cameraDetectionFailed('Camera detection timed out.');
        return;
      } on CameraException catch (error) {
        _cameraDetectionFailed(cameraErrorMessage(error));
        return;
      } catch (_) {
        _cameraDetectionFailed('The camera could not be detected.');
        return;
      }

      _detecting = false;
      if (_cameras.isEmpty) {
        _statusMessage = 'No camera detected';
        _errorMessage = 'Connect a camera and allow camera access.';
        _scheduleRetry();
        _notify();
        return;
      }

      final selected = _selectedCameraIndex;
      if (selected == null || selected < 0 || selected >= _cameras.length) {
        _selectedCameraIndex = _preferredCameraIndex(_cameras);
      }
    }

    await _initializeCamera(_selectedCameraIndex ?? 0);
  }

  void _cameraDetectionFailed(String message) {
    _cameras = const [];
    _detecting = false;
    _statusMessage = 'Camera detection failed';
    _errorMessage = message;
    _scheduleRetry();
    _notify();
  }

  Future<void> selectCamera(int index) async {
    if (index < 0 || index >= _cameras.length) return;
    if (_selectedCameraIndex == index && isReady) return;
    await _initializeCamera(index);
  }

  Future<void> _initializeCamera(int index) async {
    if (_shutdown || _suspended || index < 0 || index >= _cameras.length) {
      return;
    }

    _focusLocked = false;

    _initializing = true;
    _selectedCameraIndex = index;
    _statusMessage = 'Opening ${cameraLabel(_cameras[index], index)}...';
    _errorMessage = null;
    _notify();

    await _disposeController();

    final presets = <ResolutionPreset>[
      ResolutionPreset.high,
      ResolutionPreset.medium,
    ];

    for (final preset in presets) {
      if (_shutdown || _suspended) return;
      final opened = await _tryOpenCamera(index, preset);
      if (opened) return;
    }

    _initializing = false;
    _statusMessage = 'Camera could not open';
    _errorMessage = 'Close other camera apps and try again.';
    _scheduleRetry();
    _notify();
  }

  Future<bool> _tryOpenCamera(
    int index,
    ResolutionPreset preset,
  ) async {
    final next = CameraController(
      _cameras[index],
      preset,
      enableAudio: false,
    );
    _controller = next;
    next.addListener(_onControllerChanged);

    try {
      await next.initialize().timeout(const Duration(seconds: 20));
      if (_shutdown || _suspended || _controller != next) {
        next.removeListener(_onControllerChanged);
        await next.dispose();
        return false;
      }

      await _configureCamera(next);
      _initializing = false;
      _statusMessage = 'Camera ready';
      _errorMessage = null;
      final previewSize = next.value.previewSize;
      _resolutionStatus = previewSize == null
          ? _presetLabel(preset)
          : '${previewSize.width.round()} × ${previewSize.height.round()}';
      _retryTimer?.cancel();
      _retryTimer = null;
      _notify();

      return true;
    } catch (_) {
      next.removeListener(_onControllerChanged);
      try {
        await next.dispose();
      } catch (_) {}
      if (_controller == next) _controller = null;
      return false;
    }
  }

  Future<void> _configureCamera(CameraController current) async {
    if (Platform.isWindows) {
      final index = _selectedCameraIndex;
      if (index != null && index >= 0 && index < _cameras.length) {
        final enabled = await WindowsCameraFocusService.startAutoFocus(
          _cameras[index].name,
        ).timeout(const Duration(seconds: 5), onTimeout: () => false);
        _autofocusStatus =
            enabled ? 'Automatic — place lit egg' : 'Camera driver';
      }
      return;
    }

    var autofocusConfigured = false;
    try {
      await current.setFocusMode(FocusMode.auto);
      autofocusConfigured = true;
    } catch (_) {}

    try {
      if (current.value.focusPointSupported) {
        await current.setFocusPoint(const Offset(0.5, 0.5));
        autofocusConfigured = true;
      }
    } catch (_) {}

    if (autofocusConfigured) {
      await Future<void>.delayed(const Duration(milliseconds: 700));
      try {
        await current.setFocusMode(FocusMode.locked);
        _focusLocked = true;
      } catch (_) {}
    }

    try {
      await current.setExposureMode(ExposureMode.auto);
    } catch (_) {}

    try {
      if (current.value.exposurePointSupported) {
        await current.setExposurePoint(const Offset(0.5, 0.5));
      }
    } catch (_) {}

    _autofocusStatus = _focusLocked
        ? 'Locked on egg'
        : autofocusConfigured
            ? 'Automatic'
            : 'Camera driver';
  }

  /// Runs one autofocus sweep with the egg centered, then locks the resulting
  /// lens position. This is more stable than continuous AF at a fixed station.
  Future<bool> focusForEgg() {
    return _runFocusOperation(_focusForEggInternal);
  }

  Future<bool> focusOnIlluminatedEgg(FocusFrameScorer scoreFrame) {
    return _runFocusOperation(
      () => _focusOnIlluminatedEggInternal(scoreFrame),
    );
  }

  Future<bool> _runFocusOperation(
    Future<bool> Function() action,
  ) async {
    final running = _focusOperation;
    if (running != null) return running;

    late final Future<bool> operation;
    operation = action();
    _focusOperation = operation;

    try {
      return await operation;
    } finally {
      if (identical(_focusOperation, operation)) _focusOperation = null;
      _notify();
    }
  }

  Future<bool> _focusOnIlluminatedEggInternal(
    FocusFrameScorer scoreFrame,
  ) async {
    if (!Platform.isWindows) return _focusForEggInternal();

    final current = _controller;
    final index = _selectedCameraIndex;
    if (current == null ||
        !current.value.isInitialized ||
        index == null ||
        index < 0 ||
        index >= _cameras.length) {
      return false;
    }

    final cameraName = _cameras[index].name;
    const coarsePositions = <double>[
      0.0,
      1 / 6,
      2 / 6,
      3 / 6,
      4 / 6,
      5 / 6,
      1.0,
    ];
    var bestPosition = 0.5;
    var bestScore = double.negativeInfinity;
    Object? firstScoreError;
    StackTrace? firstScoreStackTrace;
    var sampleNumber = 0;
    final sampledPositions = <int>{};

    Future<double> captureFocusScore() async {
      XFile? frame;
      try {
        frame = await current.takePicture();
        try {
          if (current.value.isPreviewPaused) await current.resumePreview();
        } catch (_) {}
        return await scoreFrame(frame);
      } finally {
        final path = frame?.path;
        if (path != null && path.isNotEmpty) {
          try {
            await File(path).delete();
          } catch (_) {}
        }
      }
    }

    // Do not sweep the lens when the camera loads onto an empty station. The
    // current autofocus frame is enough for the backend to confirm that an
    // illuminated egg is present before manual calibration begins.
    _autofocusStatus = 'Locating illuminated egg...';
    _notify();
    try {
      await captureFocusScore();
    } catch (error, stackTrace) {
      await WindowsCameraFocusService.startAutoFocus(cameraName);
      _autofocusStatus = 'Place illuminated egg';
      Error.throwWithStackTrace(error, stackTrace);
    }

    Future<bool> sample(double position) async {
      if (_shutdown || _suspended || _controller != current) return false;

      final normalizedPosition = position.clamp(0.0, 1.0);
      final positionKey = (normalizedPosition * 1000000).round();
      if (!sampledPositions.add(positionKey)) return true;

      sampleNumber++;
      _autofocusStatus = 'Fine focus scan $sampleNumber/13';
      _notify();

      final positioned = await WindowsCameraFocusService.setManualFocusPosition(
        cameraName,
        normalizedPosition,
      ).timeout(const Duration(seconds: 5), onTimeout: () => false);
      if (!positioned) return false;

      await Future<void>.delayed(const Duration(milliseconds: 320));
      if (_shutdown || _suspended || _controller != current) return false;

      try {
        final score = await captureFocusScore();
        if (score > bestScore) {
          bestScore = score;
          bestPosition = normalizedPosition;
        }
      } catch (error, stackTrace) {
        firstScoreError ??= error;
        firstScoreStackTrace ??= stackTrace;
      }
      return true;
    }

    for (final position in coarsePositions) {
      if (!await sample(position)) {
        _autofocusStatus = 'Manual focus unavailable';
        return false;
      }
    }

    if (!bestScore.isFinite) {
      await WindowsCameraFocusService.startAutoFocus(cameraName);
      _autofocusStatus = 'No illuminated egg found';
      if (firstScoreError != null && firstScoreStackTrace != null) {
        Error.throwWithStackTrace(firstScoreError!, firstScoreStackTrace!);
      }
      return false;
    }

    // Repeatedly halve the search interval around the current winner. The
    // final 1/48 spacing is close to one focus-driver step on a C525, while the
    // initial full-range scan prevents convergence on the wrong local peak.
    for (final refinement in <double>[1 / 12, 1 / 24, 1 / 48]) {
      final centerPosition = bestPosition;
      for (final position in <double>[
        centerPosition - refinement,
        centerPosition + refinement,
      ]) {
        if (!await sample(position)) {
          _autofocusStatus = 'Manual focus unavailable';
          return false;
        }
      }
    }

    final locked = await WindowsCameraFocusService.setManualFocusPosition(
      cameraName,
      bestPosition,
    ).timeout(const Duration(seconds: 5), onTimeout: () => false);
    if (locked) {
      await Future<void>.delayed(const Duration(milliseconds: 450));
    }
    _focusLocked = locked;
    _autofocusStatus = locked ? 'Focused on illuminated egg' : 'Focus failed';
    return locked;
  }

  Future<bool> _focusForEggInternal() async {
    final current = _controller;
    if (current == null || !current.value.isInitialized) return false;

    _focusLocked = false;
    _autofocusStatus = 'Focusing on egg...';
    _notify();

    if (Platform.isWindows) {
      final index = _selectedCameraIndex;
      if (index == null || index < 0 || index >= _cameras.length) {
        _autofocusStatus = 'Camera driver';
        return false;
      }

      final cameraName = _cameras[index].name;
      final started = await WindowsCameraFocusService.startAutoFocus(cameraName)
          .timeout(const Duration(seconds: 5), onTimeout: () => false);

      if (!started) {
        _autofocusStatus = 'Camera driver';
        return false;
      }

      // Close-focus webcams can take longer to settle than phone cameras.
      await Future<void>.delayed(const Duration(milliseconds: 1600));
      if (_shutdown || _suspended || _controller != current) return false;

      final locked =
          await WindowsCameraFocusService.lockCurrentFocus(cameraName)
              .timeout(const Duration(seconds: 5), onTimeout: () => false);
      _focusLocked = locked;
      _autofocusStatus = locked ? 'Locked on egg' : 'Automatic';
      return true;
    }

    var focused = false;
    try {
      await current.setFocusMode(FocusMode.auto);
      focused = true;
    } catch (_) {}

    try {
      if (current.value.focusPointSupported) {
        await current.setFocusPoint(const Offset(0.5, 0.5));
        focused = true;
      }
    } catch (_) {}

    if (!focused) {
      _autofocusStatus = 'Camera driver';
      return false;
    }

    await Future<void>.delayed(const Duration(milliseconds: 700));
    if (_shutdown || _suspended || _controller != current) return false;
    try {
      await current.setFocusMode(FocusMode.locked);
      _focusLocked = true;
    } catch (_) {}

    _autofocusStatus = _focusLocked ? 'Locked on egg' : 'Automatic';
    return true;
  }

  Future<XFile> takePicture() async {
    await ensurePreview();
    final current = _controller;
    if (current == null || !current.value.isInitialized) {
      throw StateError('Camera unavailable');
    }

    // Reuse the calibrated lock. Refocusing for every capture can move the
    // lens toward the background immediately before the shutter fires.
    if (!_focusLocked) await focusForEgg();
    final image = await current.takePicture();

    try {
      if (current.value.isPreviewPaused) await current.resumePreview();
    } catch (_) {}

    return image;
  }

  Future<void> ensurePreview() async {
    if (_shutdown || _suspended) return;
    final current = _controller;
    if (current != null &&
        current.value.isInitialized &&
        !current.value.hasError) {
      try {
        if (current.value.isPreviewPaused) await current.resumePreview();
      } catch (_) {
        await _restartCurrentCamera();
        return;
      }
      _statusMessage = 'Camera ready';
      _errorMessage = null;
      _notify();
      return;
    }

    await start(forceRescan: _cameras.isEmpty);
  }

  Future<void> _restartCurrentCamera() async {
    if (_shutdown || _suspended) return;
    final index = _selectedCameraIndex ?? 0;
    await _initializeCamera(index);
  }

  void _onControllerChanged() {
    final current = _controller;
    if (current == null) return;
    if (current.value.hasError) {
      _statusMessage = 'Camera preview interrupted';
      _errorMessage = 'The camera preview stopped. Reconnecting...';
      _scheduleRetry();
    }
    _notify();
  }

  void _scheduleRetry() {
    if (_shutdown || _suspended || _retryTimer != null) return;
    _retryTimer = Timer.periodic(const Duration(seconds: 4), (_) {
      if (_shutdown || _suspended) return;
      if (isReady) {
        _retryTimer?.cancel();
        _retryTimer = null;
        return;
      }
      unawaited(start(forceRescan: _cameras.isEmpty));
    });
  }

  Future<void> _releaseForPause() async {
    _retryTimer?.cancel();
    _retryTimer = null;
    _focusLocked = false;
    _detecting = false;
    _initializing = false;
    _statusMessage = 'Camera paused';
    _errorMessage = null;
    await _disposeController();
    _notify();
  }

  Future<void> _disposeController() async {
    final current = _controller;
    _controller = null;
    if (current == null) return;
    current.removeListener(_onControllerChanged);
    try {
      await current.dispose();
    } catch (_) {}
  }

  Future<void> shutdown() async {
    if (_shutdown) return;
    _shutdown = true;
    _retryTimer?.cancel();
    _retryTimer = null;
    _focusLocked = false;
    WidgetsBinding.instance.removeObserver(this);
    await _disposeController();
  }

  int _preferredCameraIndex(List<CameraDescription> values) {
    final external = values.indexWhere(
      (camera) => camera.lensDirection == CameraLensDirection.external,
    );
    if (external >= 0) return external;

    final back = values.indexWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
    );
    if (back >= 0) return back;
    return 0;
  }

  String cameraLabel(CameraDescription camera, int index) {
    final raw = camera.name.trim();
    final name = raw.isEmpty ? 'Camera ${index + 1}' : raw;
    final compact = name.replaceAll(RegExp(r'\s+'), ' ').trim();
    final shortened =
        compact.length > 34 ? '${compact.substring(0, 31)}...' : compact;
    return '$shortened (${_lensLabel(camera.lensDirection)})';
  }

  String _lensLabel(CameraLensDirection direction) {
    switch (direction) {
      case CameraLensDirection.front:
        return 'front';
      case CameraLensDirection.back:
        return 'back';
      case CameraLensDirection.external:
        return 'external';
    }
  }

  String _presetLabel(ResolutionPreset preset) {
    switch (preset) {
      case ResolutionPreset.max:
        return 'Maximum available';
      case ResolutionPreset.ultraHigh:
        return 'Ultra high';
      case ResolutionPreset.veryHigh:
        return 'Very high';
      case ResolutionPreset.high:
        return 'High';
      case ResolutionPreset.medium:
        return 'Medium';
      case ResolutionPreset.low:
        return 'Low';
    }
  }

  void _notify() {
    if (!_shutdown) notifyListeners();
  }
}
