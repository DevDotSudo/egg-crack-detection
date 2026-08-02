import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../../core/constants/api_constants.dart';
import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/di/service_locator.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../services/api_service.dart';
import '../../../../shared/widgets/crack_divider.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final ApiService _api = getIt<ApiService>();

  List<Map<String, dynamic>> _items = const [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final response = await _api.dio.get(
        ApiConstants.detections,
        queryParameters: {'limit': 1000},
      );
      final data = response.data as Map;
      final items = (data['items'] as List? ?? const [])
          .whereType<Map>()
          .map((e) => e.cast<String, dynamic>())
          .toList();
      if (!mounted) return;
      setState(() {
        _items = items;
        _loading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  // ── Stats computed from the loaded items ──────────────────────────────────

  int get _total => _items.length;

  int get _cracks =>
      _items.where((i) => i['is_crack'] as bool? ?? false).length;

  String get _crackRate => _total == 0
      ? '—'
      : '${(_cracks / _total * 100).toStringAsFixed(1)}%';

  String get _avgTime {
    final times = _items
        .map((i) => i['processing_time_ms'])
        .whereType<num>()
        .toList();
    if (times.isEmpty) return '—';
    final avg = times.reduce((a, b) => a + b) / times.length;
    return '${avg.toStringAsFixed(0)} ms';
  }

  List<Map<String, dynamic>> get _recent => _items.take(5).toList();

  int _countToday() {
    final today = DateTime.now();
    return _items.where((i) {
      try {
        final ts = DateTime.parse(i['timestamp'] as String? ?? '');
        return ts.year == today.year &&
            ts.month == today.month &&
            ts.day == today.day;
      } catch (_) {
        return false;
      }
    }).length;
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final crossAxisCount =
        Responsive.value(context, desktop: 3, tablet: 2, mobile: 1);

    return Scaffold(
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _load,
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(Responsive.spaceLg),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('INSPECTION CONSOLE',
                              style: AppTheme.display(24)),
                          const SizedBox(height: 2),
                          Text(
                            'Mamdani fuzzy grading · SQLite detection store',
                            style: TextStyle(color: AppColors.shellMuted),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      onPressed: _loading ? null : _load,
                      icon: _loading
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.refresh),
                      tooltip: 'Refresh',
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                const CrackDivider(),
                const SizedBox(height: Responsive.spaceXl),

                // Feature cards
                Text('SYSTEM INFO',
                    style: AppTheme.mono(12, color: AppColors.shellFaint)),
                const SizedBox(height: Responsive.spaceMd),
                GridView.count(
                  crossAxisCount: crossAxisCount,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  mainAxisSpacing: Responsive.spaceMd,
                  crossAxisSpacing: Responsive.spaceMd,
                  childAspectRatio: 1.65,
                  children: const [
                    _InfoCard(
                      title: 'Live Preview',
                      subtitle:
                          'Flutter keeps the preview active; captured frames are scanned until no new validated crack region remains.',
                      icon: Icons.videocam_outlined,
                      color: AppColors.sage,
                    ),
                    _InfoCard(
                      title: 'Paper Pipeline',
                      subtitle:
                          'Red/green channels, Gaussian blur, binary masking, morphology, and contour scoring per Purahong et al.',
                      icon: Icons.auto_awesome_mosaic_outlined,
                      color: AppColors.amber,
                    ),
                    _InfoCard(
                      title: 'Mamdani FIS',
                      subtitle:
                          'Crack size and egg size classified via trapezoid MFs, 12-rule Mamdani FIS, and centroid-of-area defuzzification.',
                      icon: Icons.rule_folder_outlined,
                      color: AppColors.rust,
                    ),
                  ],
                ),
                const SizedBox(height: Responsive.spaceXl),

                // Live telemetry
                Text("TODAY'S TELEMETRY",
                    style: AppTheme.mono(12, color: AppColors.shellFaint)),
                const SizedBox(height: Responsive.spaceMd),
                Container(
                  padding: const EdgeInsets.all(Responsive.spaceLg),
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    border: Border.all(color: AppColors.hairline),
                  ),
                  child: Row(
                    children: [
                      _SummaryStat(
                          label: 'DETECTIONS TODAY',
                          value: _loading ? '…' : '${_countToday()}'),
                      _SummaryStat(
                          label: 'CRACK RATE',
                          value: _loading ? '…' : _crackRate),
                      _SummaryStat(
                          label: 'AVG PROCESSING TIME',
                          value: _loading ? '…' : _avgTime),
                    ],
                  ),
                ),
                const SizedBox(height: Responsive.spaceXl),

                // All-time summary strip
                Text('ALL-TIME SUMMARY',
                    style: AppTheme.mono(12, color: AppColors.shellFaint)),
                const SizedBox(height: Responsive.spaceMd),
                Row(
                  children: [
                    _QuickStat(
                        label: 'TOTAL',
                        value: _loading ? '…' : '$_total',
                        color: AppColors.shell),
                    const SizedBox(width: Responsive.spaceMd),
                    _QuickStat(
                        label: 'CRACKS',
                        value: _loading ? '…' : '$_cracks',
                        color: AppColors.rust),
                    const SizedBox(width: Responsive.spaceMd),
                    _QuickStat(
                        label: 'CLEAN',
                        value: _loading ? '…' : '${_total - _cracks}',
                        color: AppColors.sage),
                  ],
                ),
                const SizedBox(height: Responsive.spaceXl),

                // Recent results
                Text('RECENT RESULTS',
                    style: AppTheme.mono(12, color: AppColors.shellFaint)),
                const SizedBox(height: Responsive.spaceMd),
                if (_loading)
                  const Center(child: CircularProgressIndicator())
                else if (_recent.isEmpty)
                  Container(
                    padding: const EdgeInsets.all(Responsive.spaceXl),
                    alignment: Alignment.center,
                    decoration:
                        BoxDecoration(border: Border.all(color: AppColors.hairline)),
                    child: Text(
                      'No detections yet. Run your first detection to see results here.',
                      style: TextStyle(color: AppColors.shellFaint),
                    ),
                  )
                else
                  Column(
                    children: _recent.map((item) {
                      final isCrack = item['is_crack'] as bool? ?? false;
                      final source = item['source_name'] as String? ?? 'camera';
                      final eggSize = item['egg_size'] as String? ?? '—';
                      final crackSize = item['crack_size'] as String? ?? 'none';
                      final conf = (item['confidence'] as num?)?.toDouble() ?? 0;
                      final ts = item['timestamp'] as String? ?? '';
                      final color =
                          isCrack ? AppColors.rust : AppColors.sage;
                      return Container(
                        margin: const EdgeInsets.only(bottom: 6),
                        padding: const EdgeInsets.symmetric(
                            horizontal: Responsive.spaceMd, vertical: 10),
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          border: Border.all(color: AppColors.hairline),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              isCrack
                                  ? Icons.warning_amber_outlined
                                  : Icons.verified_outlined,
                              color: color,
                              size: 18,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(source,
                                      style: GoogleFonts.inter(
                                          fontSize: 13,
                                          color: AppColors.shell),
                                      overflow: TextOverflow.ellipsis),
                                  Text(
                                    '${isCrack ? "CRACK" : "CLEAN"}'
                                    ' · egg: ${eggSize.toUpperCase()}'
                                    ' · crack: ${crackSize.toUpperCase()}',
                                    style: AppTheme.mono(9.5,
                                        color: AppColors.shellFaint),
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ],
                              ),
                            ),
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.end,
                              children: [
                                Text(
                                  '${(conf * 100).toStringAsFixed(1)}%',
                                  style: AppTheme.mono(12, color: AppColors.shell),
                                ),
                                if (ts.isNotEmpty)
                                  Text(
                                    _shortTime(ts),
                                    style: AppTheme.mono(9,
                                        color: AppColors.shellFaint),
                                  ),
                              ],
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  String _shortTime(String raw) {
    try {
      final dt = DateTime.parse(raw).toLocal();
      String pad(int n) => n.toString().padLeft(2, '0');
      return '${pad(dt.hour)}:${pad(dt.minute)}';
    } catch (_) {
      return '';
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Static widgets
// ─────────────────────────────────────────────────────────────────────────────

class _InfoCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;

  const _InfoCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Responsive.spaceLg),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border.all(color: AppColors.hairline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 24),
          const Spacer(),
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: TextStyle(color: AppColors.shellMuted, fontSize: 13),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

class _SummaryStat extends StatelessWidget {
  final String label;
  final String value;

  const _SummaryStat({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value, style: AppTheme.mono(24, color: AppColors.amber)),
          const SizedBox(height: 4),
          Text(label, style: AppTheme.mono(10, color: AppColors.shellFaint)),
        ],
      ),
    );
  }
}

class _QuickStat extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _QuickStat(
      {required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(Responsive.spaceMd),
        decoration: BoxDecoration(
          color: AppColors.surface,
          border: Border.all(color: AppColors.hairline),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(value, style: AppTheme.mono(22, color: color)),
            const SizedBox(height: 4),
            Text(label, style: AppTheme.mono(9, color: AppColors.shellFaint)),
          ],
        ),
      ),
    );
  }
}
