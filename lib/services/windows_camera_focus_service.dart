import 'dart:io';

import 'package:flutter/services.dart';

class WindowsCameraFocusService {
  static const MethodChannel _channel = MethodChannel(
    'egg_camera_focus',
  );

  static Future<bool> startAutoFocus(String cameraName) {
    return _invokeFocusMethod('startAutoFocus', cameraName);
  }

  static Future<bool> lockCurrentFocus(String cameraName) {
    return _invokeFocusMethod('lockCurrentFocus', cameraName);
  }

  static Future<bool> setManualFocusPosition(
    String cameraName,
    double position,
  ) async {
    if (!Platform.isWindows) return false;

    try {
      return await _channel.invokeMethod<bool>(
            'setManualFocusPosition',
            {
              'cameraName': cameraName,
              'position': position.clamp(0.0, 1.0),
            },
          ) ??
          false;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> _invokeFocusMethod(
    String method,
    String cameraName,
  ) async {
    if (!Platform.isWindows) return true;

    try {
      return await _channel.invokeMethod<bool>(
            method,
            {'cameraName': cameraName},
          ) ??
          false;
    } catch (_) {
      return false;
    }
  }
}
