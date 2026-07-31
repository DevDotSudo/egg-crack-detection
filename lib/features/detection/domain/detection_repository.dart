import 'dart:typed_data';
import 'detection_result.dart';

abstract class DetectionRepository {
  /// Sends a single image to the backend `/detect` endpoint.
  Future<DetectionResult> detectSingle({
    required Uint8List imageBytes,
    required String filename,
    bool includeIntermediateSteps = false,
  });
}
