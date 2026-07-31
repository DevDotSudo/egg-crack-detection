import 'package:flutter/material.dart';

/// Inspection-console palette.
///
/// Grounded in the paper's own visual world rather than a generic Material
/// seed color: the dark box + flashlight rig the eggs are photographed in
/// (near-black warm ground, amber key light), and the red/green channel
/// split the pipeline actually runs on (rust = crack/alert, sage = pass).
class AppColors {
  AppColors._();

  // Ground — the "dark box" the app lives in.
  static const Color ink = Color(0xFF14120D);
  static const Color surface = Color(0xFF1F1B15);
  static const Color surfaceRaised = Color(0xFF272219);
  static const Color hairline = Color(0xFF3A3327);

  // Text — eggshell, not paper-white.
  static const Color shell = Color(0xFFF3E9D4);
  static const Color shellMuted = Color(0xFFA89C86);
  static const Color shellFaint = Color(0xFF6E6552);

  // Signal — tied directly to the pipeline's channel split.
  static const Color rust = Color(0xFFC1440E); // crack / alert
  static const Color rustDim = Color(0xFF3A2015);
  static const Color sage = Color(0xFF74936B); // pass / verified
  static const Color sageDim = Color(0xFF23281F);
  static const Color amber = Color(0xFFD9A441); // accent, focus, scan-line

  // Legacy aliases kept for widgets not yet migrated.
  static const Color background = ink;
  static const Color border = hairline;
  static const Color textPrimary = shell;
  static const Color textSecondary = shellMuted;
  static const Color textDisabled = shellFaint;
  static const Color danger = rust;
  static const Color dangerLight = rustDim;
  static const Color success = sage;
  static const Color successLight = sageDim;
  static const Color warning = amber;
  static const Color primary = amber;
  static const Color primaryDark = Color(0xFFB5872F);

  static const Color crackBadgeBg = rustDim;
  static const Color crackBadgeText = Color(0xFFE07A4C);
  static const Color noCrackBadgeBg = sageDim;
  static const Color noCrackBadgeText = Color(0xFF9BBF92);
}
