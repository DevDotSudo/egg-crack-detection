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
import '../../../detection/domain/detection_result.dart';

class BatchProcessingScreen extends StatefulWidget {
  const BatchProcessingScreen({super.key});

  @override
  State<BatchProcessingScreen> createState() => _BatchProcessingScreenState();
}

class _BatchProcessingScreenState extends State<BatchProcessingScreen> {
  final ApiService _api = getIt<ApiService>();
  final List<_BatchItem> _items = [];
  bool _processing = false;
  String? _message;

  static const _imageTypes = XTypeGroup(
    label: 'images',
    extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'],
  );

  Future<void> _selectFiles() async {
    final files = await openFiles(acceptedTypeGroups: [_imageTypes]);
    if (files.isEmpty) return;
    setState(() {
      _message = null;
      _items
        ..clear()
        ..addAll(files.map(_BatchItem.fromXFile));
    });
  }

  Future<void> _selectFolder() async {
    final directoryPath = await getDirectoryPath();
    if (directoryPath == null) return;
    final directory = Directory(directoryPath);
    final files = directory
        .listSync()
        .whereType<File>()
        .where((file) => _isImage(file.path))
        .map((file) => XFile(file.path))
        .toList();
    setState(() {
      _message = files.isEmpty ? 'No supported images found in that folder.' : null;
      _items
        ..clear()
        ..addAll(files.map(_BatchItem.fromXFile));
    });
  }

  Future<void> _processQueue() async {
    if (_processing || _items.isEmpty) return;
    setState(() {
      _processing = true;
      _message = null;
      for (final item in _items) {
        item.status = _BatchStatus.queued;
        item.result = null;
        item.error = null;
      }
    });

    for (final item in _items) {
      if (!mounted) return;
      setState(() => item.status = _BatchStatus.processing);
      try {
        final bytes = await item.file.readAsBytes();
        final formData = FormData.fromMap({
          'image': MultipartFile.fromBytes(bytes, filename: item.name),
          'include_intermediate_steps': 'false',
        });
        final response = await _api.dio.post(
          ApiConstants.detect,
          data: formData,
        );
        item.result =
            DetectionResult.fromJson(response.data as Map<String, dynamic>);
        item.status = _BatchStatus.done;
      } catch (error) {
        item.error = friendlyErrorMessage(
          error,
          fallback: 'This image could not be analyzed.',
        );
        item.status = _BatchStatus.failed;
      }
      if (mounted) setState(() {});
    }

    if (!mounted) return;
    setState(() {
      _processing = false;
      _message = 'Batch complete.';
    });
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

  bool _isImage(String path) {
    final lower = path.toLowerCase();
    return lower.endsWith('.jpg') ||
        lower.endsWith('.jpeg') ||
        lower.endsWith('.png') ||
        lower.endsWith('.bmp') ||
        lower.endsWith('.webp');
  }

  @override
  Widget build(BuildContext context) {
    final done = _items.where((item) => item.status == _BatchStatus.done).length;
    final failed =
        _items.where((item) => item.status == _BatchStatus.failed).length;

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Responsive.spaceLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('BATCH PROCESSING', style: AppTheme.display(22)),
              const SizedBox(height: 2),
              Text(
                'Run detection across a folder or selected image files.',
                style: TextStyle(color: AppColors.shellMuted),
              ),
              const SizedBox(height: 12),
              const CrackDivider(seed: 5),
              const SizedBox(height: Responsive.spaceLg),
              Wrap(
                spacing: Responsive.spaceMd,
                runSpacing: Responsive.spaceSm,
                children: [
                  OutlinedButton.icon(
                    onPressed: _processing ? null : _selectFolder,
                    icon: const Icon(Icons.folder_open_outlined),
                    label: const Text('Select folder'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _processing ? null : _selectFiles,
                    icon: const Icon(Icons.file_copy_outlined),
                    label: const Text('Select files'),
                  ),
                  ElevatedButton.icon(
                    onPressed:
                        _processing || _items.isEmpty ? null : _processQueue,
                    icon: const Icon(Icons.play_arrow_outlined),
                    label: const Text('Run batch'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _exportCsv,
                    icon: const Icon(Icons.download_outlined),
                    label: const Text('Export CSV'),
                  ),
                ],
              ),
              const SizedBox(height: Responsive.spaceMd),
              Text(
                '${_items.length} queued  |  $done done  |  $failed failed',
                style: AppTheme.mono(11, color: AppColors.shellFaint),
              ),
              if (_message != null) ...[
                const SizedBox(height: 8),
                Text(_message!, style: TextStyle(color: AppColors.shellMuted)),
              ],
              const SizedBox(height: Responsive.spaceLg),
              Expanded(
                child: _items.isEmpty
                    ? Container(
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          border: Border.all(color: AppColors.hairline),
                        ),
                        child: Text(
                          'No images queued. Select a folder or multiple files to begin.',
                          style: TextStyle(color: AppColors.shellFaint),
                        ),
                      )
                    : ListView.separated(
                        itemCount: _items.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 8),
                        itemBuilder: (context, index) {
                          return _BatchTile(item: _items[index]);
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

class _BatchTile extends StatelessWidget {
  final _BatchItem item;

  const _BatchTile({required this.item});

  @override
  Widget build(BuildContext context) {
    final result = item.result;
    final color = switch (item.status) {
      _BatchStatus.done => result?.isCrack == true ? AppColors.rust : AppColors.sage,
      _BatchStatus.failed => AppColors.rust,
      _BatchStatus.processing => AppColors.amber,
      _BatchStatus.queued => AppColors.shellFaint,
    };
    final statusText = switch (item.status) {
      _BatchStatus.done => result?.isCrack == true
          ? 'Crack - ${result?.crackSize.toUpperCase()}'
          : 'No crack',
      _BatchStatus.failed => 'Failed',
      _BatchStatus.processing => 'Processing',
      _BatchStatus.queued => 'Queued',
    };

    return Container(
      padding: const EdgeInsets.all(Responsive.spaceMd),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border.all(color: AppColors.hairline),
      ),
      child: Row(
        children: [
          Icon(Icons.image_outlined, color: color),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(item.name, overflow: TextOverflow.ellipsis),
                const SizedBox(height: 4),
                Text(
                  item.error ?? statusText,
                  style: AppTheme.mono(11, color: AppColors.shellFaint),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          if (result != null)
            Text(
              '${(result.confidence * 100).toStringAsFixed(1)}%',
              style: AppTheme.mono(12, color: AppColors.shell),
            ),
        ],
      ),
    );
  }
}

class _BatchItem {
  final XFile file;
  final String name;
  _BatchStatus status = _BatchStatus.queued;
  DetectionResult? result;
  String? error;

  _BatchItem({
    required this.file,
    required this.name,
  });

  factory _BatchItem.fromXFile(XFile file) {
    final path = file.path;
    final normalized = path.replaceAll('\\', '/');
    final name = normalized.split('/').last;
    return _BatchItem(file: file, name: name.isEmpty ? file.name : name);
  }
}

enum _BatchStatus { queued, processing, done, failed }
