import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../../core/constants/api_constants.dart';
import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/di/service_locator.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../services/api_service.dart';
import '../../../../services/error_message_service.dart';
import '../../../../shared/widgets/crack_divider.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Screen
// ─────────────────────────────────────────────────────────────────────────────

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

  // ── Data loading ────────────────────────────────────────────────────────────

  Future<void> _loadHistory() async {
    setState(() {
      _loading = true;
      _errorMessage = null;
    });
    try {
      final response = await _api.dio.get(
        ApiConstants.detections,
        queryParameters: {'limit': 200},
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
      await _api.dio.delete(ApiConstants.detectionById(id));
      if (!mounted) return;
      setState(() {
        _items = _items.where((item) => item['id'] != id).toList();
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage = friendlyErrorMessage(
          error,
          fallback: 'The record could not be deleted.',
        );
      });
    }
  }

  Future<void> _clearHistory() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: Text('Clear all history?', style: AppTheme.display(16)),
        content: Text(
          'This will permanently delete all detection records and their saved images.',
          style: TextStyle(color: AppColors.shellMuted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: AppColors.rust),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete all'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    try {
      await _api.dio.delete(ApiConstants.detections);
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

  // ── Image popup ─────────────────────────────────────────────────────────────

  void _showDetailDialog(Map<String, dynamic> item) {
    showDialog<void>(
      context: context,
      barrierColor: Colors.black87,
      builder: (ctx) => _DetectionDetailDialog(
        item: item,
        api: _api,
        onDelete: () {
          final id = item['id'] as String?;
          if (id != null) {
            Navigator.of(ctx).pop();
            _deleteItem(id);
          }
        },
      ),
    );
  }

  // ── Build ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Responsive.spaceLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header row
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('HISTORY', style: AppTheme.display(22)),
                        const SizedBox(height: 2),
                        Text(
                          '${_items.length} detection${_items.length == 1 ? '' : 's'} · tap a row to view image',
                          style: TextStyle(color: AppColors.shellMuted, fontSize: 12),
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
                    onPressed: _loading || _items.isEmpty ? null : _clearHistory,
                    icon: const Icon(Icons.delete_sweep_outlined),
                    tooltip: 'Clear all history',
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const CrackDivider(seed: 9),
              const SizedBox(height: Responsive.spaceLg),

              // Error banner
              if (_errorMessage != null) ...[
                Container(
                  padding: const EdgeInsets.all(Responsive.spaceMd),
                  decoration: BoxDecoration(
                    color: AppColors.rustDim,
                    border: Border.all(color: AppColors.rust.withValues(alpha: 0.4)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.warning_amber_outlined,
                          color: AppColors.rust, size: 16),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _errorMessage!,
                          style: TextStyle(color: AppColors.rust, fontSize: 12),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: Responsive.spaceMd),
              ],

              // List
              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : _items.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: const [
                                Icon(Icons.history_outlined,
                                    size: 48, color: AppColors.shellFaint),
                                SizedBox(height: 12),
                                Text(
                                  'No detection history yet.',
                                  style: TextStyle(color: AppColors.shellFaint),
                                ),
                              ],
                            ),
                          )
                        : ListView.separated(
                            itemCount: _items.length,
                            separatorBuilder: (_, __) =>
                                const SizedBox(height: 6),
                            itemBuilder: (context, index) {
                              final item = _items[index];
                              return _HistoryTile(
                                item: item,
                                onTap: () => _showDetailDialog(item),
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

// ─────────────────────────────────────────────────────────────────────────────
// History tile
// ─────────────────────────────────────────────────────────────────────────────

class _HistoryTile extends StatefulWidget {
  final Map<String, dynamic> item;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const _HistoryTile({
    required this.item,
    required this.onTap,
    required this.onDelete,
  });

  @override
  State<_HistoryTile> createState() => _HistoryTileState();
}

class _HistoryTileState extends State<_HistoryTile> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final item = widget.item;
    final isCrack = item['is_crack'] as bool? ?? false;
    final confidence = _double(item['confidence']);
    final crackSize = item['crack_size'] as String? ?? 'none';
    final eggSize = item['egg_size'] as String? ?? 'unknown';
    final processingTime = (item['processing_time_ms'] as num? ?? 0).toInt();
    final timestamp = item['timestamp'] as String? ?? '';
    final source = item['source_name'] as String? ?? 'camera';
    final hasImage = item['original_image_path'] != null;

    final signalColor = isCrack ? AppColors.rust : AppColors.sage;
    final badgeBg = isCrack ? AppColors.crackBadgeBg : AppColors.noCrackBadgeBg;
    final badgeText = isCrack ? AppColors.crackBadgeText : AppColors.noCrackBadgeText;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          padding: const EdgeInsets.symmetric(
              horizontal: Responsive.spaceMd, vertical: 12),
          decoration: BoxDecoration(
            color: _hovered ? AppColors.surfaceRaised : AppColors.surface,
            border: Border.all(
              color: _hovered
                  ? AppColors.amber.withValues(alpha: 0.5)
                  : AppColors.hairline,
            ),
          ),
          child: Row(
            children: [
              // Status icon
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: badgeBg,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Icon(
                  isCrack
                      ? Icons.warning_amber_outlined
                      : Icons.verified_outlined,
                  color: signalColor,
                  size: 18,
                ),
              ),
              const SizedBox(width: 12),

              // Main info
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            source,
                            overflow: TextOverflow.ellipsis,
                            style: Theme.of(context).textTheme.titleSmall,
                          ),
                        ),
                        if (hasImage)
                          Padding(
                            padding: const EdgeInsets.only(left: 6),
                            child: Icon(
                              Icons.image_outlined,
                              size: 14,
                              color: AppColors.amber.withValues(alpha: 0.7),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '${isCrack ? '⚠ CRACK' : '✓ CLEAN'}'
                      '  ·  crack: ${crackSize.toUpperCase()}'
                      '  ·  egg: ${eggSize.toUpperCase()}'
                      '  ·  ${processingTime}ms',
                      style: AppTheme.mono(10, color: AppColors.shellFaint),
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (timestamp.isNotEmpty) ...[
                      const SizedBox(height: 2),
                      Text(
                        _formatTimestamp(timestamp),
                        style: AppTheme.mono(9.5, color: AppColors.shellFaint),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ],
                ),
              ),
              const SizedBox(width: 12),

              // Confidence badge
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: badgeBg,
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  '${(confidence * 100).toStringAsFixed(1)}%',
                  style: AppTheme.mono(11, color: badgeText),
                ),
              ),
              const SizedBox(width: 4),

              // Delete
              IconButton(
                onPressed: widget.onDelete,
                icon: const Icon(Icons.delete_outline, size: 16),
                tooltip: 'Delete',
                color: AppColors.shellFaint,
                splashRadius: 18,
              ),
            ],
          ),
        ),
      ),
    );
  }

  double _double(Object? value) => value is num ? value.toDouble() : 0;

  String _formatTimestamp(String raw) {
    try {
      final dt = DateTime.parse(raw).toLocal();
      String pad(int n) => n.toString().padLeft(2, '0');
      return '${dt.year}-${pad(dt.month)}-${pad(dt.day)}'
          '  ${pad(dt.hour)}:${pad(dt.minute)}:${pad(dt.second)}';
    } catch (_) {
      return raw;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detail popup dialog
// ─────────────────────────────────────────────────────────────────────────────

class _DetectionDetailDialog extends StatefulWidget {
  final Map<String, dynamic> item;
  final ApiService api;
  final VoidCallback onDelete;

  const _DetectionDetailDialog({
    required this.item,
    required this.api,
    required this.onDelete,
  });

  @override
  State<_DetectionDetailDialog> createState() => _DetectionDetailDialogState();
}

class _DetectionDetailDialogState extends State<_DetectionDetailDialog>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;
  // 0 = original, 1 = overlay
  final List<Uint8List?> _imageData = [null, null];
  final List<bool> _loadingImage = [false, false];
  final List<String?> _imageError = [null, null];

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);
    _fetchImage(0);
    _fetchImage(1);
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  Future<void> _fetchImage(int index) async {
    final id = widget.item['id'] as String?;
    if (id == null) return;

    setState(() {
      _loadingImage[index] = true;
      _imageError[index] = null;
    });

    final url = index == 0
        ? ApiConstants.detectionOriginalImage(id)
        : ApiConstants.detectionOverlayImage(id);

    try {
      final response = await widget.api.dio.get<List<int>>(
        url,
        options: Options(responseType: ResponseType.bytes),
      );
      if (!mounted) return;
      setState(() {
        _imageData[index] = Uint8List.fromList(response.data!);
        _loadingImage[index] = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loadingImage[index] = false;
        _imageError[index] = 'Image not available';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final item = widget.item;
    final isCrack = item['is_crack'] as bool? ?? false;
    final confidence = _double(item['confidence']);
    final eggSize = item['egg_size'] as String? ?? 'unknown';
    final crackSize = item['crack_size'] as String? ?? 'none';
    final eggSizeConf = _double(item['egg_size_confidence']);
    final crackSizeConf = _double(item['crack_size_confidence']);
    final source = item['source_name'] as String? ?? 'camera';
    final timestamp = item['timestamp'] as String? ?? '';
    final processingTime = (item['processing_time_ms'] as num? ?? 0).toInt();
    final contourLength = _double(item['contour_length']);
    final eggWidthPx = _double(item['egg_width_pixels']);
    final eggLengthPx = _double(item['egg_length_pixels']);
    final memberships = item['egg_size_memberships'] as Map? ?? {};

    final signalColor = isCrack ? AppColors.rust : AppColors.sage;

    final screenSize = MediaQuery.of(context).size;
    final dialogWidth = (screenSize.width * 0.82).clamp(500.0, 960.0);
    final dialogHeight = (screenSize.height * 0.86).clamp(480.0, 760.0);

    return Dialog(
      backgroundColor: AppColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(6),
        side: BorderSide(color: AppColors.hairline),
      ),
      child: SizedBox(
        width: dialogWidth,
        height: dialogHeight,
        child: Column(
          children: [
            // ── Dialog header ──────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: Responsive.spaceLg, vertical: 14),
              decoration: BoxDecoration(
                color: AppColors.surfaceRaised,
                border: Border(bottom: BorderSide(color: AppColors.hairline)),
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(6)),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: isCrack
                          ? AppColors.crackBadgeBg
                          : AppColors.noCrackBadgeBg,
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: Text(
                      isCrack ? '⚠  CRACK DETECTED' : '✓  NO CRACK',
                      style: AppTheme.mono(11,
                          color: isCrack
                              ? AppColors.crackBadgeText
                              : AppColors.noCrackBadgeText),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      source,
                      overflow: TextOverflow.ellipsis,
                      style: AppTheme.display(14),
                    ),
                  ),
                  IconButton(
                    onPressed: widget.onDelete,
                    icon: const Icon(Icons.delete_outline, size: 18),
                    tooltip: 'Delete this record',
                    color: AppColors.rust,
                    splashRadius: 18,
                  ),
                  IconButton(
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.close, size: 20),
                    splashRadius: 18,
                    color: AppColors.shellMuted,
                  ),
                ],
              ),
            ),

            // ── Body: image left, stats right ─────────────────────────────
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Image pane
                  Expanded(
                    flex: 6,
                    child: Column(
                      children: [
                        // Tab bar
                        Container(
                          color: AppColors.ink,
                          child: TabBar(
                            controller: _tabs,
                            indicatorColor: AppColors.amber,
                            labelColor: AppColors.amber,
                            unselectedLabelColor: AppColors.shellFaint,
                            labelStyle:
                                AppTheme.mono(11, color: AppColors.amber),
                            unselectedLabelStyle: AppTheme.mono(11,
                                color: AppColors.shellFaint),
                            tabs: const [
                              Tab(text: 'ORIGINAL'),
                              Tab(text: 'OVERLAY'),
                            ],
                          ),
                        ),
                        // Image view
                        Expanded(
                          child: TabBarView(
                            controller: _tabs,
                            children: [
                              _ImagePane(
                                data: _imageData[0],
                                loading: _loadingImage[0],
                                error: _imageError[0],
                                onRetry: () => _fetchImage(0),
                              ),
                              _ImagePane(
                                data: _imageData[1],
                                loading: _loadingImage[1],
                                error: _imageError[1],
                                onRetry: () => _fetchImage(1),
                                overlayTint: isCrack,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Divider
                  Container(
                      width: 1, color: AppColors.hairline),

                  // Stats pane
                  SizedBox(
                    width: 220,
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.all(Responsive.spaceMd),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _StatSection('DETECTION'),
                          _StatRow('Result',
                              isCrack ? 'Crack' : 'No crack', signalColor),
                          _StatRow('Confidence',
                              '${(confidence * 100).toStringAsFixed(1)}%',
                              AppColors.shell),
                          _StatRow('Crack size',
                              crackSize.toUpperCase(), AppColors.shellMuted),
                          _StatRow('Crack size conf.',
                              '${(crackSizeConf * 100).toStringAsFixed(1)}%',
                              AppColors.shellMuted),
                          if (contourLength > 0)
                            _StatRow('Contour length',
                                '${contourLength.toStringAsFixed(1)} px',
                                AppColors.shellMuted),
                          const SizedBox(height: 12),
                          _StatSection('EGG SIZE'),
                          _StatRow('Size', eggSize.toUpperCase(),
                              AppColors.amber),
                          _StatRow('Confidence',
                              '${(eggSizeConf * 100).toStringAsFixed(1)}%',
                              AppColors.shellMuted),
                          if (eggWidthPx > 0)
                            _StatRow('Width',
                                '${eggWidthPx.toStringAsFixed(1)} px',
                                AppColors.shellMuted),
                          if (eggLengthPx > 0)
                            _StatRow('Length',
                                '${eggLengthPx.toStringAsFixed(1)} px',
                                AppColors.shellMuted),
                          // Mamdani memberships bar chart
                          if (memberships.isNotEmpty) ...[
                            const SizedBox(height: 8),
                            Text('Memberships',
                                style: AppTheme.mono(9.5,
                                    color: AppColors.shellFaint)),
                            const SizedBox(height: 4),
                            for (final entry in ['small', 'medium', 'large'])
                              _MembershipBar(
                                label: entry,
                                value: _double(memberships[entry]),
                              ),
                          ],
                          const SizedBox(height: 12),
                          _StatSection('CAPTURE'),
                          _StatRow('Processing',
                              '${processingTime}ms', AppColors.shellMuted),
                          if (timestamp.isNotEmpty)
                            _StatRow('Time',
                                _formatTimestamp(timestamp),
                                AppColors.shellFaint),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  double _double(Object? v) => v is num ? v.toDouble() : 0.0;

  String _formatTimestamp(String raw) {
    try {
      final dt = DateTime.parse(raw).toLocal();
      String pad(int n) => n.toString().padLeft(2, '0');
      return '${dt.year}-${pad(dt.month)}-${pad(dt.day)}\n'
          '${pad(dt.hour)}:${pad(dt.minute)}:${pad(dt.second)}';
    } catch (_) {
      return raw;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image pane
// ─────────────────────────────────────────────────────────────────────────────

class _ImagePane extends StatelessWidget {
  final Uint8List? data;
  final bool loading;
  final String? error;
  final VoidCallback onRetry;
  final bool overlayTint;

  const _ImagePane({
    required this.data,
    required this.loading,
    required this.error,
    required this.onRetry,
    this.overlayTint = false,
  });

  @override
  Widget build(BuildContext context) {
    if (loading) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(strokeWidth: 2),
            SizedBox(height: 12),
            Text('Loading image…',
                style: TextStyle(color: AppColors.shellFaint, fontSize: 12)),
          ],
        ),
      );
    }

    if (error != null || data == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.image_not_supported_outlined,
                color: AppColors.shellFaint, size: 40),
            const SizedBox(height: 8),
            Text(
              error ?? 'No image available',
              style: const TextStyle(
                  color: AppColors.shellFaint, fontSize: 12),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh, size: 14),
              label: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    return InteractiveViewer(
      minScale: 0.5,
      maxScale: 8.0,
      child: Center(
        child: ColorFiltered(
          colorFilter: overlayTint
              ? const ColorFilter.matrix(<double>[
                  1.05, 0, 0, 0, 0,
                  0, 0.95, 0, 0, 0,
                  0, 0, 0.95, 0, 0,
                  0, 0, 0, 1, 0,
                ])
              : const ColorFilter.matrix(<double>[
                  1, 0, 0, 0, 0,
                  0, 1, 0, 0, 0,
                  0, 0, 1, 0, 0,
                  0, 0, 0, 1, 0,
                ]),
          child: Image.memory(
            data!,
            fit: BoxFit.contain,
            gaplessPlayback: true,
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small stat widgets
// ─────────────────────────────────────────────────────────────────────────────

class _StatSection extends StatelessWidget {
  final String label;
  const _StatSection(this.label);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Text(
        label,
        style: GoogleFonts.jetBrainsMono(
          fontSize: 9,
          fontWeight: FontWeight.w700,
          color: AppColors.amber,
          letterSpacing: 1.2,
        ),
      ),
    );
  }
}

class _StatRow extends StatelessWidget {
  final String label;
  final String value;
  final Color valueColor;

  const _StatRow(this.label, this.value, this.valueColor);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 90,
            child: Text(label,
                style: AppTheme.mono(9.5, color: AppColors.shellFaint)),
          ),
          Expanded(
            child: Text(value,
                style: AppTheme.mono(9.5, color: valueColor),
                overflow: TextOverflow.ellipsis),
          ),
        ],
      ),
    );
  }
}

class _MembershipBar extends StatelessWidget {
  final String label;
  final double value; // 0.0 – 1.0

  const _MembershipBar({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final pct = value.clamp(0.0, 1.0);
    final barColor = switch (label) {
      'small' => AppColors.sage,
      'medium' => AppColors.amber,
      'large' => AppColors.rust,
      _ => AppColors.shellFaint,
    };

    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          SizedBox(
            width: 46,
            child: Text(label,
                style: AppTheme.mono(9, color: AppColors.shellFaint)),
          ),
          Expanded(
            child: Stack(
              children: [
                Container(
                  height: 6,
                  decoration: BoxDecoration(
                    color: AppColors.ink,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: pct,
                  child: Container(
                    height: 6,
                    decoration: BoxDecoration(
                      color: barColor.withValues(alpha: 0.75),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 6),
          SizedBox(
            width: 34,
            child: Text(
              '${(pct * 100).toStringAsFixed(0)}%',
              style: AppTheme.mono(9, color: AppColors.shellFaint),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}
