import 'package:flutter/material.dart';

import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/responsive.dart';
import '../../../../core/theme/app_theme.dart';
import '../../../../shared/widgets/crack_divider.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final crossAxisCount =
        Responsive.value(context, desktop: 3, tablet: 2, mobile: 1);

    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(Responsive.spaceLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('INSPECTION CONSOLE', style: AppTheme.display(24)),
              const SizedBox(height: 2),
              Text(
                'Egg crack detection based on Purahong et al. plus fuzzy grading.',
                style: TextStyle(color: AppColors.shellMuted),
              ),
              const SizedBox(height: 14),
              const CrackDivider(),
              const SizedBox(height: Responsive.spaceXl),
              Text(
                'SYSTEM INFO',
                style: AppTheme.mono(12, color: AppColors.shellFaint),
              ),
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
                        'Flutter keeps the preview active; one captured image is scanned repeatedly until no new validated crack region remains.',
                    icon: Icons.videocam_outlined,
                    color: AppColors.sage,
                  ),
                  _InfoCard(
                    title: 'Paper Pipeline',
                    subtitle:
                        'Red and green channels, 11x11 Gaussian blur, binary masking, morphology, and contour scoring.',
                    icon: Icons.auto_awesome_mosaic_outlined,
                    color: AppColors.amber,
                  ),
                  _InfoCard(
                    title: 'Fuzzy Grading',
                    subtitle:
                        'Crack size and apparent egg size are classified as small, medium, or large with confidence.',
                    icon: Icons.rule_folder_outlined,
                    color: AppColors.rust,
                  ),
                ],
              ),
              const SizedBox(height: Responsive.spaceXl),
              Text(
                'TODAY\'S TELEMETRY',
                style: AppTheme.mono(12, color: AppColors.shellFaint),
              ),
              const SizedBox(height: Responsive.spaceMd),
              Container(
                padding: const EdgeInsets.all(Responsive.spaceLg),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  border: Border.all(color: AppColors.hairline),
                ),
                child: const Row(
                  children: [
                    _SummaryStat(label: 'DETECTIONS TODAY', value: '0'),
                    _SummaryStat(label: 'CRACK RATE', value: '0%'),
                    _SummaryStat(label: 'AVG PROCESSING TIME', value: '-'),
                  ],
                ),
              ),
              const SizedBox(height: Responsive.spaceXl),
              Text(
                'RECENT RESULTS',
                style: AppTheme.mono(12, color: AppColors.shellFaint),
              ),
              const SizedBox(height: Responsive.spaceMd),
              Container(
                padding: const EdgeInsets.all(Responsive.spaceXl),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  border: Border.all(color: AppColors.hairline),
                ),
                child: Text(
                  'No detections yet. Run your first detection to see results here.',
                  style: TextStyle(color: AppColors.shellFaint),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

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
