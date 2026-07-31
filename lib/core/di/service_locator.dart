import 'package:get_it/get_it.dart';
import '../../features/camera/data/camera_repository_impl.dart';
import '../../features/camera/domain/camera_repository.dart';
import '../../features/detection/data/detection_repository_impl.dart';
import '../../features/detection/domain/detection_repository.dart';
import '../../features/detection/presentation/cubit/detection_cubit.dart';
import '../../services/api_service.dart';
import '../../services/backend_launcher_service.dart';
import '../../services/camera_session_service.dart';
import '../../services/file_service.dart';

final getIt = GetIt.instance;

void setupServiceLocator() {
  getIt.registerLazySingleton<BackendLauncherService>(
    () => BackendLauncherService(),
  );
  getIt.registerLazySingleton<ApiService>(
    () => ApiService(getIt<BackendLauncherService>()),
  );
  getIt.registerLazySingleton<FileService>(() => FileService());
  getIt.registerLazySingleton<CameraSessionService>(
    () => CameraSessionService(),
  );

  getIt.registerLazySingleton<DetectionRepository>(
    () => DetectionRepositoryImpl(getIt<ApiService>()),
  );
  getIt.registerLazySingleton<CameraRepository>(
    () => CameraRepositoryImpl(getIt<ApiService>()),
  );

  getIt.registerFactory<DetectionCubit>(
    () => DetectionCubit(getIt<DetectionRepository>()),
  );
}
