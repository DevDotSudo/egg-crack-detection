/// Central place for backend connection details and endpoint paths.
/// Keep this in sync with `shared/api_contract.md`.
class ApiConstants {
  ApiConstants._();

  static const String host = '127.0.0.1';
  static const int port = 8756;
  static const String baseUrl = 'http://$host:$port';

  static const Duration connectTimeout = Duration(seconds: 5);
  static const Duration receiveTimeout = Duration(seconds: 30);

  // Health
  static const String health = '/health';

  // Detection
  static const String detect = '/detect';
  static const String detectCamera = '/detect/camera';
  static const String detectBatch = '/detect/batch';
  static const String detectCameraMulti = '/detect/camera/multi';
  static const String focusScore = '/focus/score';

  // History
  static const String history = '/history';
  static String historyById(String id) => '/history/$id';

  // Reports
  static const String reportsExport = '/reports/export';
}
