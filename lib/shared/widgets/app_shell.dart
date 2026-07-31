import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/responsive.dart';
import '../../core/theme/app_theme.dart';

class NavDestinationItem {
  final String label;
  final IconData icon;
  final IconData selectedIcon;
  final String route;

  const NavDestinationItem({
    required this.label,
    required this.icon,
    required this.selectedIcon,
    required this.route,
  });
}

const List<NavDestinationItem> kNavItems = [
  NavDestinationItem(
    label: 'Dashboard',
    icon: Icons.dashboard_outlined,
    selectedIcon: Icons.dashboard,
    route: '/',
  ),
  NavDestinationItem(
    label: 'Detect',
    icon: Icons.image_search_outlined,
    selectedIcon: Icons.image_search,
    route: '/detection',
  ),
  NavDestinationItem(
    label: 'Camera',
    icon: Icons.videocam_outlined,
    selectedIcon: Icons.videocam,
    route: '/camera',
  ),
  NavDestinationItem(
    label: 'Batch',
    icon: Icons.dynamic_feed_outlined,
    selectedIcon: Icons.dynamic_feed,
    route: '/batch',
  ),
  NavDestinationItem(
    label: 'History',
    icon: Icons.history_outlined,
    selectedIcon: Icons.history,
    route: '/history',
  ),
  NavDestinationItem(
    label: 'Reports',
    icon: Icons.bar_chart_outlined,
    selectedIcon: Icons.bar_chart,
    route: '/reports',
  ),
];

/// Slim, icon-first console rail -- reads as an instrument panel, not a
/// generic app drawer. Labels sit under the icon rather than beside it,
/// so the rail stays narrow even on desktop.
class AppShell extends StatelessWidget {
  final Widget child;
  final String currentRoute;
  final void Function(String route) onNavigate;

  const AppShell({
    super.key,
    required this.child,
    required this.currentRoute,
    required this.onNavigate,
  });

  int get _selectedIndex {
    final index = kNavItems.indexWhere((item) => item.route == currentRoute);
    return index == -1 ? 0 : index;
  }

  @override
  Widget build(BuildContext context) {
    final isDesktop = Responsive.isDesktop(context);

    if (isDesktop) {
      return Scaffold(
        body: Row(
          children: [
            Container(
              width: 84,
              color: AppColors.surface,
              child: Column(
                children: [
                  const SizedBox(height: 20),
                  Text('EGG', style: AppTheme.mono(11, color: AppColors.amber)),
                  Text('CRACK', style: AppTheme.mono(11, color: AppColors.shellFaint)),
                  const SizedBox(height: 20),
                  const Divider(height: 1, indent: 20, endIndent: 20),
                  const SizedBox(height: 12),
                  for (var i = 0; i < kNavItems.length; i++)
                    _RailTile(
                      item: kNavItems[i],
                      selected: i == _selectedIndex,
                      onTap: () => onNavigate(kNavItems[i].route),
                    ),
                ],
              ),
            ),
            const VerticalDivider(width: 1),
            Expanded(child: child),
          ],
        ),
      );
    }

    // Tablet / mobile fallback
    return Scaffold(
      body: child,
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => onNavigate(kNavItems[i].route),
        destinations: kNavItems
            .map(
              (item) => NavigationDestination(
                icon: Icon(item.icon),
                selectedIcon: Icon(item.selectedIcon),
                label: item.label,
              ),
            )
            .toList(),
      ),
    );
  }
}

class _RailTile extends StatelessWidget {
  final NavDestinationItem item;
  final bool selected;
  final VoidCallback onTap;

  const _RailTile({
    required this.item,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = selected ? AppColors.amber : AppColors.shellFaint;

    return InkWell(
      onTap: onTap,
      child: Container(
        width: 84,
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          border: Border(
            left: BorderSide(
              color: selected ? AppColors.amber : Colors.transparent,
              width: 2,
            ),
          ),
        ),
        child: Column(
          children: [
            Icon(selected ? item.selectedIcon : item.icon, color: color, size: 22),
            const SizedBox(height: 4),
            Text(
              item.label,
              style: AppTheme.mono(9.5, color: color),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
