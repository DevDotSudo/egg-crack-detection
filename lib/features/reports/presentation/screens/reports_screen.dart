import 'dart:io';

import 'package:dio/dio.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import '../../../../core/constants/api_constants.dart';
import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/di/service_locator.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../services/api_service.dart';
import '../../../../services/error_message_service.dart';
import '../../../../shared/widgets/crack_divider.dart';

class ReportsScreen extends StatefulWidget {
  const ReportsScreen({super.key});

  @override
  State<ReportsScreen> createState() => _ReportsScreenState();
}

class _ReportsScreenState extends State<ReportsScreen> {
  final ApiService _api = getIt<ApiService>();
  List<Map<String, dynamic>> _items = const [];
  bool _loading = true;
  String? _message;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _message = null;
    });

    try {
      final response = await _api.dio.get(
        ApiConstants.detections,
        queryParameters: {'limit': 1000},
      );
      final data = response.data as Map;
      final items = (data['items'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => item.cast<String, dynamic>())
          .toList();
      if (!mounted) return;
      setState(() {
        _items = items;
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _message = friendlyErrorMessage(
          error,
          fallback: 'Report data could not be loaded.',
        );
      });
    }
  }

  Future<void> _exportCsv() async {
    final location = await getSaveLocation(
      suggestedName: 'egg_crack_report.csv',
      acceptedTypeGroups: const [
        XTypeGroup(label: 'CSV', extensions: ['csv']),
      ],
    );
    if (location == null) return;

    try {
      final response = await _api.dio.get<List<int>>(
        ApiConstants.reportsExport,
        options: Options(responseType: ResponseType.bytes),
      );
      await File(location.path).writeAsBytes(response.data ?? const []);
      if (!mounted) return;
      setState(() => _message = 'CSV exported to ${location.path}');
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _message = friendlyErrorMessage(
          error,
          fallback: 'The CSV report could not be exported.',
        );
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final total = _items.length;
    final cracks =
        _items.where((item) => item['is_crack'] as bool? ?? false).length;
    final noCracks = total - cracks;
    final crackRate = total == 0 ? 0.0 : cracks / total;
    final times = _items
        .map((item) => item['processing_time_ms'])
        .whereType<num>()
        .map((value) => value.toDouble())
        .toList();
    final avgTime = times.isEmpty
        ? 0.0
        : times.reduce((a, b) => a + b) / times.length;

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Responsive.spaceLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('REPORTS', style: AppTheme.display(22)),
                        const SizedBox(height: 2),
                        Text(
                          'Backend detection summary and CSV export.',
                          style: TextStyle(color: AppColors.shellMuted),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: _loading ? null : _load,
                    icon: const Icon(Icons.refresh),
                    tooltip: 'Refresh',
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const CrackDivider(seed: 11),
              const SizedBox(height: Responsive.spaceLg),
              if (_message != null) ...[
                Text(_message!, style: TextStyle(color: AppColors.shellMuted)),
                const SizedBox(height: Responsive.spaceMd),
              ],
              if (_loading)
                const Expanded(child: Center(child: CircularProgressIndicator()))
              else
                Expanded(
                  child: Column(
                    children: [
                      GridView.count(
                        crossAxisCount: Responsive.value(
                          context,
                          desktop: 4,
                          tablet: 2,
                          mobile: 1,
                        ),
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        mainAxisSpacing: Responsive.spaceMd,
                        crossAxisSpacing: Responsive.spaceMd,
                        childAspectRatio: 1.9,
                        children: [
                          _MetricCard(label: 'TOTAL', value: '$total'),
                          _MetricCard(label: 'CRACK', value: '$cracks'),
                          _MetricCard(label: 'NO CRACK', value: '$noCracks'),
                          _MetricCard(
                            label: 'CRACK RATE',
                            value: '${(crackRate * 100).toStringAsFixed(1)}%',
                          ),
                        ],
                      ),
                      const SizedBox(height: Responsive.spaceLg),
                      Container(
                        padding: const EdgeInsets.all(Responsive.spaceLg),
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          border: Border.all(color: AppColors.hairline),
                        ),
                        child: Column(
                          children: [
                            _ReportRow(
                              label: 'AVERAGE PROCESSING TIME',
                              value: '${avgTime.toStringAsFixed(0)} ms',
                            ),
                            _ReportRow(
                              label: 'SMALL CRACKS',
                              value: '${_countCrackSize('small')}',
                            ),
                            _ReportRow(
                              label: 'MEDIUM CRACKS',
                              value: '${_countCrackSize('medium')}',
                            ),
                            _ReportRow(
                              label: 'LARGE CRACKS',
                              value: '${_countCrackSize('large')}',
                            ),
                            const Divider(),
                            _ReportRow(
                              label: 'SMALL EGGS',
                              value: '${_countEggSize('small')}',
                            ),
                            _ReportRow(
                              label: 'MEDIUM EGGS',
                              value: '${_countEggSize('medium')}',
                            ),
                            _ReportRow(
                              label: 'LARGE EGGS',
                              value: '${_countEggSize('large')}',
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: Responsive.spaceLg),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: ElevatedButton.icon(
                          onPressed: _exportCsv,
                          icon: const Icon(Icons.download_outlined),
                          label: const Text('Export CSV summary'),
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  int _countCrackSize(String size) {
    return _items
        .where((item) => (item['crack_size'] as String? ?? '') == size)
        .length;
  }

  int _countEggSize(String size) {
    return _items
        .where((item) => (item['egg_size'] as String? ?? '') == size)
        .length;
  }
}

class _MetricCard extends StatelessWidget {
  final String label;
  final String value;

  const _MetricCard({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Responsive.spaceMd),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border.all(color: AppColors.hairline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(value, style: AppTheme.mono(24, color: AppColors.amber)),
          const SizedBox(height: 4),
          Text(label, style: AppTheme.mono(10, color: AppColors.shellFaint)),
        ],
      ),
    );
  }
}

class _ReportRow extends StatelessWidget {
  final String label;
  final String value;

  const _ReportRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: AppTheme.mono(11, color: AppColors.shellFaint)),
          Text(value, style: AppTheme.mono(12, color: AppColors.shell)),
        ],
      ),
    );
  }
}
