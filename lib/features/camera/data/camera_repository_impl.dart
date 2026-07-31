import 'dart:typed_data';

import 'package:dio/dio.dart';

import '../../../core/constants/api_constants.dart';
import '../../../services/api_service.dart';
import '../../detection/domain/detection_result.dart';
import '../domain/camera_repository.dart';

class CameraRepositoryImpl implements CameraRepository {
  final ApiService _api;

  CameraRepositoryImpl(this._api);

  @override
  Future<DetectionResult> detectCapturedImage({
    required Uint8List imageBytes,
    required String filename,
    bool includeIntermediateSteps = false,
  }) async {
    final formData = FormData.fromMap({
      'image': MultipartFile.fromBytes(imageBytes, filename: filename),
      'include_intermediate_steps': includeIntermediateSteps.toString(),
    });

    final response = await _api.dio.post(
      ApiConstants.detectCamera,
      data: formData,
      options: Options(receiveTimeout: const Duration(seconds: 120)),
    );

    return DetectionResult.fromJson(response.data as Map<String, dynamic>);
  }

  @override
  Future<DetectionResult> detectMultiFrame({
    required List<Uint8List> frames,
    required String filename,
  }) async {
    final formData = FormData();
    for (var i = 0; i < frames.length; i++) {
      formData.files.add(MapEntry(
        'images',
        MultipartFile.fromBytes(frames[i], filename: '${i}_$filename'),
      ));
    }

    final response = await _api.dio.post(
      ApiConstants.detectCameraMulti,
      data: formData,
      options: Options(receiveTimeout: const Duration(seconds: 120)),
    );

    return DetectionResult.fromJson(response.data as Map<String, dynamic>);
  }

  @override
  Future<double> scoreFocusFrame({
    required Uint8List imageBytes,
    required String filename,
  }) async {
    final response = await _api.dio.post(
      ApiConstants.focusScore,
      data: FormData.fromMap({
        'image': MultipartFile.fromBytes(imageBytes, filename: filename),
      }),
      options: Options(receiveTimeout: const Duration(seconds: 30)),
    );

    final data = response.data as Map<String, dynamic>;
    return (data['focus_score'] as num).toDouble();
  }

  @override
  Future<void> saveDetection({
    required DetectionResult result,
    required String sourceName,
  }) async {
    await _api.dio.post(
      ApiConstants.history,
      data: {
        'result': result.toJson(),
        'source_name': sourceName,
      },
    );
  }
}
