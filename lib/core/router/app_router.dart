import 'package:go_router/go_router.dart';
import '../../features/batch_processing/presentation/screens/batch_processing_screen.dart';
import '../../features/camera/presentation/screens/camera_screen.dart';
import '../../features/dashboard/presentation/screens/dashboard_screen.dart';
import '../../features/detection/presentation/screens/detection_screen.dart';
import '../../features/history/presentation/screens/history_screen.dart';
import '../../features/reports/presentation/screens/reports_screen.dart';
import '../../shared/widgets/app_shell.dart';

/// Single ShellRoute wraps every top-level screen so the nav rail /
/// bottom nav persists across navigation instead of rebuilding.
final appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    ShellRoute(
      builder: (context, state, child) {
        return AppShell(
          currentRoute: state.matchedLocation,
          onNavigate: (route) => context.go(route),
          child: child,
        );
      },
      routes: [
        GoRoute(
          path: '/',
          builder: (context, state) => const DashboardScreen(),
        ),
        GoRoute(
          path: '/detection',
          builder: (context, state) => const DetectionScreen(),
        ),
        GoRoute(
          path: '/camera',
          builder: (context, state) => const CameraScreen(),
        ),
        GoRoute(
          path: '/batch',
          builder: (context, state) => const BatchProcessingScreen(),
        ),
        GoRoute(
          path: '/history',
          builder: (context, state) => const HistoryScreen(),
        ),
        GoRoute(
          path: '/reports',
          builder: (context, state) => const ReportsScreen(),
        ),
      ],
    ),
  ],
);
