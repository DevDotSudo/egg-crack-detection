import 'dart:typed_data';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../../../services/error_message_service.dart';
import '../../domain/detection_repository.dart';
import 'detection_state.dart';

class DetectionCubit extends Cubit<DetectionState> {
  final DetectionRepository _repository;

  DetectionCubit(this._repository) : super(const DetectionState());

  void imagePicked(String filename) {
    emit(state.copyWith(
      status: DetectionStatus.imageSelected,
      selectedFilename: filename,
      errorMessage: null,
    ));
  }

  void togglePipelineSteps(bool value) {
    emit(state.copyWith(showPipelineSteps: value));
  }

  Future<void> runDetection(Uint8List imageBytes, String filename) async {
    emit(state.copyWith(status: DetectionStatus.loading));
    try {
      final result = await _repository.detectSingle(
        imageBytes: imageBytes,
        filename: filename,
        includeIntermediateSteps: state.showPipelineSteps,
      );
      emit(state.copyWith(status: DetectionStatus.success, result: result));
    } catch (e) {
      emit(state.copyWith(
        status: DetectionStatus.failure,
        errorMessage: friendlyErrorMessage(
          e,
          fallback: 'The image could not be analyzed.',
        ),
      ));
    }
  }

  void reset() => emit(const DetectionState());
}
