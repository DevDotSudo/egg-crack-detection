import 'package:equatable/equatable.dart';
import '../../domain/detection_result.dart';

enum DetectionStatus { initial, imageSelected, loading, success, failure }

class DetectionState extends Equatable {
  final DetectionStatus status;
  final String? selectedFilename;
  final DetectionResult? result;
  final String? errorMessage;
  final bool showPipelineSteps;

  const DetectionState({
    this.status = DetectionStatus.initial,
    this.selectedFilename,
    this.result,
    this.errorMessage,
    this.showPipelineSteps = false,
  });

  DetectionState copyWith({
    DetectionStatus? status,
    String? selectedFilename,
    DetectionResult? result,
    String? errorMessage,
    bool? showPipelineSteps,
  }) {
    return DetectionState(
      status: status ?? this.status,
      selectedFilename: selectedFilename ?? this.selectedFilename,
      result: result ?? this.result,
      errorMessage: errorMessage,
      showPipelineSteps: showPipelineSteps ?? this.showPipelineSteps,
    );
  }

  @override
  List<Object?> get props =>
      [status, selectedFilename, result, errorMessage, showPipelineSteps];
}
