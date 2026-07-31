import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

/// Inspection-console theme.
///
/// Type pairing is deliberate: Space Grotesk for headers and readouts
/// carries the geometric, instrument-panel feel; Inter stays for body
/// copy where plain legibility matters; JetBrains Mono is reserved for
/// numeric telemetry (contour length, ms, timestamps) so data reads as
/// *measured*, not just styled text.
class AppTheme {
  AppTheme._();

  static TextStyle display(double size, {FontWeight weight = FontWeight.w600, Color? color}) {
    return GoogleFonts.spaceGrotesk(
      fontSize: size,
      fontWeight: weight,
      color: color ?? AppColors.shell,
      letterSpacing: -0.2,
    );
  }

  static TextStyle mono(double size, {FontWeight weight = FontWeight.w500, Color? color}) {
    return GoogleFonts.jetBrainsMono(
      fontSize: size,
      fontWeight: weight,
      color: color ?? AppColors.shell,
    );
  }

  static ThemeData get dark {
    final body = GoogleFonts.interTextTheme(ThemeData.dark().textTheme);

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.ink,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.amber,
        secondary: AppColors.sage,
        error: AppColors.rust,
        surface: AppColors.surface,
        onSurface: AppColors.shell,
      ),
      textTheme: body.apply(
        bodyColor: AppColors.shell,
        displayColor: AppColors.shell,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.ink,
        foregroundColor: AppColors.shell,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        titleTextStyle: display(18, weight: FontWeight.w600),
      ),
      cardTheme: CardThemeData(
        color: AppColors.surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: const BorderSide(color: AppColors.hairline),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.amber,
          foregroundColor: AppColors.ink,
          disabledBackgroundColor: AppColors.surfaceRaised,
          disabledForegroundColor: AppColors.shellFaint,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
          textStyle: GoogleFonts.inter(fontWeight: FontWeight.w600, letterSpacing: 0.2),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.shell,
          side: const BorderSide(color: AppColors.hairline),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected) ? AppColors.amber : AppColors.shellFaint,
        ),
        trackColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected) ? AppColors.amber.withValues(alpha: 0.3) : AppColors.surfaceRaised,
        ),
      ),
      dividerTheme: const DividerThemeData(color: AppColors.hairline, thickness: 1),
      navigationRailTheme: NavigationRailThemeData(
        backgroundColor: AppColors.surface,
        indicatorColor: AppColors.amber.withValues(alpha: 0.15),
        selectedIconTheme: const IconThemeData(color: AppColors.amber),
        unselectedIconTheme: const IconThemeData(color: AppColors.shellFaint),
        selectedLabelTextStyle: GoogleFonts.inter(color: AppColors.amber, fontWeight: FontWeight.w600, fontSize: 11),
        unselectedLabelTextStyle: GoogleFonts.inter(color: AppColors.shellFaint, fontSize: 11),
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: AppColors.surface,
        indicatorColor: AppColors.amber.withValues(alpha: 0.15),
        iconTheme: WidgetStateProperty.resolveWith(
          (states) => IconThemeData(
            color: states.contains(WidgetState.selected) ? AppColors.amber : AppColors.shellFaint,
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.surfaceRaised,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(4),
          borderSide: const BorderSide(color: AppColors.hairline),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(4),
          borderSide: const BorderSide(color: AppColors.hairline),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(4),
          borderSide: const BorderSide(color: AppColors.amber),
        ),
        labelStyle: GoogleFonts.inter(color: AppColors.shellMuted),
      ),
    );
  }

  /// Retained for any screen not yet migrated to the console theme.
  static ThemeData get light => dark;
}
