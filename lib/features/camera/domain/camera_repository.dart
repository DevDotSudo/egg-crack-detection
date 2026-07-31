import 'dart:typed_data';

import '../../detection/domain/detection_result.dart';

abstract class CameraRepository {
  Future<DetectionResult> detectCapturedImage({
    required Uint8List imageBytes,
    required String filename,
    bool includeIntermediateSteps = false,
  });

  Future<DetectionResult> detectMultiFrame({
    required List<Uint8List> frames,
    required String filename,
  });

  Future<double> scoreFocusFrame({
    required Uint8List imageBytes,
    required String filename,
  });

  Future<void> saveDetection({
    required DetectionResult result,
    required String sourceName,
  });
}
