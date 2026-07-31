import 'package:flutter/material.dart';

import '../../../../core/constants/api_constants.dart';
import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/di/service_locator.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../services/api_service.dart';
import '../../../../services/error_message_service.dart';
import '../../../../shared/widgets/crack_divider.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final ApiService _api = getIt<ApiService>();
  List<Map<String, dynamic>> _items = const [];
  bool _loading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() {
      _loading = true;
      _errorMessage = null;
    });

    try {
      final response = await _api.dio.get(
        ApiConstants.history,
        queryParameters: {'limit': 100},
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
        _errorMessage = friendlyErrorMessage(
          error,
          fallback: 'History could not be loaded.',
        );
      });
    }
  }

  Future<void> _deleteItem(String id) async {
    try {
      await _api.dio.delete(ApiConstants.historyById(id));
      if (!mounted) return;
      setState(() {
        _items = _items.where((item) => item['id'] != id).toList();
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage = friendlyErrorMessage(
          error,
          fallback: 'The history item could not be deleted.',
        );
      });
    }
  }

  Future<void> _clearHistory() async {
    try {
      await _api.dio.delete(ApiConstants.history);
      if (!mounted) return;
      setState(() {
        _items = const [];
        _errorMessage = null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage = friendlyErrorMessage(
          error,
          fallback: 'History could not be cleared.',
        );
      });
    }
  }

  @override
  Widget build(BuildContext context) {
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
                        Text('HISTORY', style: AppTheme.display(22)),
                        const SizedBox(height: 2),
                        Text(
                          'Past detections from local backend storage.',
                          style: TextStyle(color: AppColors.shellMuted),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: _loading ? null : _loadHistory,
                    icon: const Icon(Icons.refresh),
                    tooltip: 'Refresh',
                  ),
                  IconButton(
                    onPressed:
                        _loading || _items.isEmpty ? null : _clearHistory,
                    icon: const Icon(Icons.delete_outline),
                    tooltip: 'Clear history',
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const CrackDivider(seed: 9),
              const SizedBox(height: Responsive.spaceLg),
              if (_errorMessage != null) ...[
                Text(
                  _errorMessage!,
                  style: TextStyle(color: AppColors.rust, fontSize: 12),
                ),
                const SizedBox(height: Responsive.spaceMd),
              ],
              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : _items.isEmpty
                        ? Center(
                            child: Text(
                              'No detection history yet.',
                              style: TextStyle(color: AppColors.shellFaint),
                            ),
                          )
                        : ListView.separated(
                            itemCount: _items.length,
                            separatorBuilder: (_, __) =>
                                const SizedBox(height: 8),
                            itemBuilder: (context, index) {
                              final item = _items[index];
                              return _HistoryTile(
                                item: item,
                                onDelete: () {
                                  final id = item['id'] as String?;
                                  if (id != null) _deleteItem(id);
                                },
                              );
                            },
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final Map<String, dynamic> item;
  final VoidCallback onDelete;

  const _HistoryTile({required this.item, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    final isCrack = item['is_crack'] as bool? ?? false;
    final confidence = _double(item['confidence']);
    final crackSize = item['crack_size'] as String? ?? 'none';
    final eggSize = item['egg_size'] as String? ?? 'unknown';
    final processingTime = (item['processing_time_ms'] as num? ?? 0).toInt();
    final timestamp = item['timestamp'] as String? ?? '';
    final source = item['source_name'] as String? ?? 'camera';
    final color = isCrack ? AppColors.rust : AppColors.sage;

    return Container(
      padding: const EdgeInsets.all(Responsive.spaceMd),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border.all(color: AppColors.hairline),
      ),
      child: Row(
        children: [
          Icon(
            isCrack ? Icons.warning_amber_outlined : Icons.verified_outlined,
            color: color,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  source,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 4),
                Text(
                  '${isCrack ? 'Crack' : 'No crack'} | crack: ${crackSize.toUpperCase()} | egg: ${eggSize.toUpperCase()} | ${processingTime}ms',
                  style: AppTheme.mono(11, color: AppColors.shellFaint),
                  overflow: TextOverflow.ellipsis,
                ),
                if (timestamp.isNotEmpty) ...[
                  const SizedBox(height: 3),
                  Text(
                    timestamp,
                    style: AppTheme.mono(10, color: AppColors.shellFaint),
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ],
            ),
          ),
          Text(
            '${(confidence * 100).toStringAsFixed(1)}%',
            style: AppTheme.mono(12, color: AppColors.shell),
          ),
          IconButton(
            onPressed: onDelete,
            icon: const Icon(Icons.delete_outline, size: 18),
            tooltip: 'Delete',
          ),
        ],
      ),
    );
  }

  double _double(Object? value) => value is num ? value.toDouble() : 0;
}
