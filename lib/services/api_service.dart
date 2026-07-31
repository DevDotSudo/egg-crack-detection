import 'dart:async';

import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import 'backend_launcher_service.dart';

class ApiService {
  final BackendLauncherService _launcher;
  late final Dio dio;
  late final Dio _healthDio;
  Future<bool>? _startupOperation;
  DateTime? _lastHealthyAt;

  ApiService(this._launcher) {
    dio = Dio(
      BaseOptions(
        baseUrl: ApiConstants.baseUrl,
        connectTimeout: ApiConstants.connectTimeout,
        receiveTimeout: ApiConstants.receiveTimeout,
      ),
    );

    _healthDio = Dio(
      BaseOptions(
        baseUrl: ApiConstants.baseUrl,
        connectTimeout: const Duration(milliseconds: 800),
        receiveTimeout: const Duration(seconds: 1),
        sendTimeout: const Duration(seconds: 1),
      ),
    );

    dio.interceptors.add(
      QueuedInterceptorsWrapper(
        onRequest: (options, handler) async {
          if (options.path == ApiConstants.health) {
            handler.next(options);
            return;
          }

          final ready = await ensureBackendReady();
          if (!ready) {
            handler.reject(
              DioException(
                requestOptions: options,
                type: DioExceptionType.connectionError,
                error: const BackendUnavailableException(),
              ),
            );
            return;
          }

          handler.next(options);
        },
      ),
    );
  }

  Future<bool> ensureBackendReady({
    Duration timeout = const Duration(seconds: 30),
  }) async {
    final lastHealthyAt = _lastHealthyAt;
    if (lastHealthyAt != null &&
        DateTime.now().difference(lastHealthyAt) < const Duration(seconds: 3)) {
      return true;
    }

    final running = _startupOperation;
    if (running != null) return running;

    late final Future<bool> operation;
    operation = _ensureBackendReadyInternal(timeout);
    _startupOperation = operation;
    try {
      return await operation;
    } finally {
      if (identical(_startupOperation, operation)) {
        _startupOperation = null;
      }
    }
  }

  Future<bool> _ensureBackendReadyInternal(Duration timeout) async {
    if (await _isHealthy()) return true;

    final launched = await _launcher.start();
    if (!launched) return _isHealthy();

    return waitUntilHealthy(timeout: timeout);
  }

  Future<bool> waitUntilHealthy({
    Duration timeout = const Duration(seconds: 20),
    Duration pollInterval = const Duration(milliseconds: 400),
  }) async {
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      if (await _isHealthy()) return true;
      await Future<void>.delayed(pollInterval);
    }
    return false;
  }

  Future<bool> _isHealthy() async {
    try {
      final response = await _healthDio.get(ApiConstants.health);
      final healthy = response.statusCode == 200;
      if (healthy) _lastHealthyAt = DateTime.now();
      return healthy;
    } catch (_) {
      return false;
    }
  }
}

class BackendUnavailableException implements Exception {
  const BackendUnavailableException();
}
