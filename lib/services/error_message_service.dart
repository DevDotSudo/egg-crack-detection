import 'package:camera/camera.dart';
import 'package:dio/dio.dart';

import 'api_service.dart';

String friendlyErrorMessage(
  Object error, {
  String fallback = 'Something went wrong. Please try again.',
}) {
  if (error is BackendUnavailableException) {
    return 'The detection service could not start. Please restart the app.';
  }

  if (error is DioException) {
    if (error.error is BackendUnavailableException) {
      return 'The detection service could not start. Please restart the app.';
    }

    switch (error.type) {
      case DioExceptionType.connectionError:
      case DioExceptionType.connectionTimeout:
        return 'The detection service is not ready. Please try again.';
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
      case DioExceptionType.transformTimeout:
        return 'Processing took too long. Please try again.';
      case DioExceptionType.badResponse:
        return _responseMessage(error.response, fallback);
      case DioExceptionType.cancel:
        return 'The request was cancelled.';
      case DioExceptionType.badCertificate:
        return 'A secure connection could not be created.';
      case DioExceptionType.unknown:
        return fallback;
    }
  }

  if (error is CameraException) {
    return cameraErrorMessage(error);
  }

  return fallback;
}

String cameraErrorMessage(CameraException error) {
  switch (error.code) {
    case 'CameraAccessDenied':
      return 'Camera permission was denied.';
    case 'CameraAccessDeniedWithoutPrompt':
      return 'Camera access is disabled in system settings.';
    case 'CameraAccessRestricted':
      return 'Camera access is restricted on this device.';
    case 'CameraInUse':
      return 'The camera is being used by another app.';
    default:
      return 'The camera could not start. Close other camera apps and try again.';
  }
}

String _responseMessage(Response<dynamic>? response, String fallback) {
  final statusCode = response?.statusCode ?? 0;
  final data = response?.data;

  if (data is Map) {
    final detail = data['detail'];
    if (detail is String && _isSafeMessage(detail)) {
      return detail;
    }
    final message = data['message'];
    if (message is String && _isSafeMessage(message)) {
      return message;
    }
  }

  if (statusCode == 400) return 'The image could not be processed.';
  if (statusCode == 404) return 'The requested data was not found.';
  if (statusCode == 413) return 'The selected image is too large.';
  if (statusCode == 415 || statusCode == 422) {
    return 'Please select a valid egg image.';
  }
  if (statusCode >= 500) {
    return 'The detector could not process the image. Please try again.';
  }

  return fallback;
}

bool _isSafeMessage(String value) {
  final text = value.trim();
  if (text.isEmpty || text.length > 180) return false;
  final lower = text.toLowerCase();
  return !lower.contains('traceback') &&
      !lower.contains('exception') &&
      !lower.contains('socketexception') &&
      !lower.contains('dioexception') &&
      !lower.contains('errno');
}
