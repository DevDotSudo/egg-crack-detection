import 'package:flutter/material.dart';

/// Breakpoints for the three layout modes the app supports.
/// This is primarily a Windows desktop app, but the shell stays adaptive
/// so the window can be resized/snapped without breaking layout.
class Responsive {
  Responsive._();

  static const double mobileMax = 600;
  static const double tabletMax = 1024;

  static bool isMobile(BuildContext context) =>
      MediaQuery.of(context).size.width < mobileMax;

  static bool isTablet(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    return width >= mobileMax && width < tabletMax;
  }

  static bool isDesktop(BuildContext context) =>
      MediaQuery.of(context).size.width >= tabletMax;

  /// Convenience picker for widgets that differ per breakpoint.
  static T value<T>(
    BuildContext context, {
    required T desktop,
    T? tablet,
    T? mobile,
  }) {
    if (isDesktop(context)) return desktop;
    if (isTablet(context)) return tablet ?? desktop;
    return mobile ?? tablet ?? desktop;
  }

  static const double spaceXs = 4;
  static const double spaceSm = 8;
  static const double spaceMd = 16;
  static const double spaceLg = 24;
  static const double spaceXl = 32;
}
