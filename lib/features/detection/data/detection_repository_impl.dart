import 'dart:typed_data';
import 'package:dio/dio.dart';
import '../../../core/constants/api_constants.dart';
import '../../../services/api_service.dart';
import '../domain/detection_repository.dart';
import '../domain/detection_result.dart';

class DetectionRepositoryImpl implements DetectionRepository {
  final ApiService _api;

  DetectionRepositoryImpl(this._api);

  @override
  Future<DetectionResult> detectSingle({
    required Uint8List imageBytes,
    required String filename,
    bool includeIntermediateSteps = false,
  }) async {
    final formData = FormData.fromMap({
      'image': MultipartFile.fromBytes(imageBytes, filename: filename),
      'include_intermediate_steps': includeIntermediateSteps.toString(),
    });

    final response = await _api.dio.post(
      ApiConstants.detect,
      data: formData,
    );

    return DetectionResult.fromJson(response.data as Map<String, dynamic>);
  }
}
