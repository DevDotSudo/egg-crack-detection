import 'package:equatable/equatable.dart';

class DetectionResult extends Equatable {
  final String? id;
  final bool isCrack;
  final double confidence;
  final double areaRatio;
  final double contourLength;
  final int processingTimeMs;
  final String originalImageB64;
  final String overlayImageB64;
  final Map<String, String>? intermediateSteps;
  final DateTime timestamp;
  final int candidateComponents;
  final int rawCandidateComponents;
  final bool dominantCrackOverride;
  final int candidatePixels;
  final double longestCandidate;
  final double meanCandidateStrength;
  final double detectionScore;
  final int thresholdUsed;
  final double shellTextureScore;
  final double shellTextureUniformity;
  final double textureAnomalyRatio;
  final int textureCandidatePixels;
  final double thinCrackScore;
  final bool thinCrackDetected;
  final bool eggDetected;
  final double eggScore;
  final String eggSize;
  final double eggSizeConfidence;
  final double eggAreaRatio;
  final double eggWidthPixels;
  final double eggLengthPixels;
  final double eggWidthRatio;
  final double eggLengthRatio;
  final double eggSizeScore;
  final Map<String, double> eggSizeMemberships;
  final String crackSize;
  final double crackSizeConfidence;
  final String crackMaskB64;
  final List<Map<String, dynamic>> crackLocations;
  final int detectionIterations;
  final int searchIterations;
  final String terminationReason;
  final int sampleCount;
  final int crackVotes;
  final int noCrackVotes;
  final double decisionConsistency;
  final bool areaConsistent;
  final double areaConsistency;
  final double areaMeanRatio;
  final double areaSpreadRatio;
  final List<double> areaSamples;

  const DetectionResult({
    this.id,
    required this.isCrack,
    required this.confidence,
    required this.areaRatio,
    required this.contourLength,
    required this.processingTimeMs,
    required this.originalImageB64,
    required this.overlayImageB64,
    this.intermediateSteps,
    required this.timestamp,
    this.candidateComponents = 0,
    this.rawCandidateComponents = 0,
    this.dominantCrackOverride = false,
    this.candidatePixels = 0,
    this.longestCandidate = 0,
    this.meanCandidateStrength = 0,
    this.detectionScore = 0,
    this.thresholdUsed = 0,
    this.shellTextureScore = 0,
    this.shellTextureUniformity = 1,
    this.textureAnomalyRatio = 0,
    this.textureCandidatePixels = 0,
    this.thinCrackScore = 0,
    this.thinCrackDetected = false,
    this.eggDetected = true,
    this.eggScore = 0,
    this.eggSize = 'unknown',
    this.eggSizeConfidence = 0,
    this.eggAreaRatio = 0,
    this.eggWidthPixels = 0,
    this.eggLengthPixels = 0,
    this.eggWidthRatio = 0,
    this.eggLengthRatio = 0,
    this.eggSizeScore = 0,
    this.eggSizeMemberships = const {},
    this.crackSize = 'none',
    this.crackSizeConfidence = 0,
    this.crackMaskB64 = '',
    this.crackLocations = const [],
    this.detectionIterations = 0,
    this.searchIterations = 1,
    this.terminationReason = 'no_more_cracks',
    this.sampleCount = 1,
    this.crackVotes = 0,
    this.noCrackVotes = 0,
    this.decisionConsistency = 1,
    this.areaConsistent = true,
    this.areaConsistency = 1,
    this.areaMeanRatio = 0,
    this.areaSpreadRatio = 0,
    this.areaSamples = const [],
  });

  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      id: json['id'] as String?,
      isCrack: json['is_crack'] as bool? ?? false,
      confidence: _double(json['confidence']),
      areaRatio: _double(json['area_ratio']),
      contourLength: _double(json['contour_length']),
      processingTimeMs: (json['processing_time_ms'] as num? ?? 0).toInt(),
      originalImageB64: json['original_image_b64'] as String? ?? '',
      overlayImageB64: json['overlay_image_b64'] as String? ?? '',
      intermediateSteps: (json['intermediate_steps'] as Map?)
          ?.map((key, value) => MapEntry(key.toString(), value.toString())),
      timestamp: json['timestamp'] != null
          ? DateTime.parse(json['timestamp'] as String)
          : DateTime.now(),
      candidateComponents:
          (json['candidate_components'] as num? ?? 0).toInt(),
      rawCandidateComponents:
          (json['raw_candidate_components'] as num? ?? 0).toInt(),
      dominantCrackOverride:
          json['dominant_crack_override'] as bool? ?? false,
      candidatePixels: (json['candidate_pixels'] as num? ?? 0).toInt(),
      longestCandidate: _double(json['longest_candidate']),
      meanCandidateStrength: _double(json['mean_candidate_strength']),
      detectionScore: _double(json['detection_score']),
      thresholdUsed: (json['threshold_used'] as num? ?? 0).toInt(),
      shellTextureScore: _double(json['shell_texture_score']),
      shellTextureUniformity: _double(
        json['shell_texture_uniformity'],
        fallback: 1,
      ),
      textureAnomalyRatio: _double(json['texture_anomaly_ratio']),
      textureCandidatePixels:
          (json['texture_candidate_pixels'] as num? ?? 0).toInt(),
      thinCrackScore: _double(json['thin_crack_score']),
      thinCrackDetected: json['thin_crack_detected'] as bool? ?? false,
      eggDetected: json['egg_detected'] as bool? ?? true,
      eggScore: _double(json['egg_score']),
      eggSize: json['egg_size'] as String? ?? 'unknown',
      eggSizeConfidence: _double(json['egg_size_confidence']),
      eggAreaRatio: _double(json['egg_area_ratio']),
      eggWidthPixels: _double(json['egg_width_pixels']),
      eggLengthPixels: _double(json['egg_length_pixels']),
      eggWidthRatio: _double(json['egg_width_ratio']),
      eggLengthRatio: _double(json['egg_length_ratio']),
      eggSizeScore: _double(json['egg_size_score']),
      eggSizeMemberships: (json['egg_size_memberships'] as Map? ?? const {})
          .map((key, value) => MapEntry(key.toString(), _double(value))),
      crackSize: json['crack_size'] as String? ?? 'none',
      crackSizeConfidence: _double(json['crack_size_confidence']),
      crackMaskB64: json['crack_mask_b64'] as String? ?? '',
      crackLocations: (json['crack_locations'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => item.map(
                (key, value) => MapEntry(key.toString(), value),
              ))
          .toList(),
      detectionIterations:
          (json['detection_iterations'] as num? ?? 0).toInt(),
      searchIterations: (json['search_iterations'] as num? ?? 1).toInt(),
      terminationReason:
          json['termination_reason'] as String? ?? 'no_more_cracks',
      sampleCount: (json['sample_count'] as num? ?? 1).toInt(),
      crackVotes: (json['crack_votes'] as num? ??
              ((json['is_crack'] as bool? ?? false) ? 1 : 0))
          .toInt(),
      noCrackVotes: (json['no_crack_votes'] as num? ??
              ((json['is_crack'] as bool? ?? false) ? 0 : 1))
          .toInt(),
      decisionConsistency: _double(
        json['decision_consistency'],
        fallback: 1,
      ),
      areaConsistent: json['area_consistent'] as bool? ?? true,
      areaConsistency: _double(json['area_consistency'], fallback: 1),
      areaMeanRatio: _double(json['area_mean_ratio']),
      areaSpreadRatio: _double(json['area_spread_ratio']),
      areaSamples: (json['area_samples'] as List? ?? const [])
          .whereType<num>()
          .map((value) => value.toDouble())
          .toList(),
    );
  }


  Map<String, dynamic> toJson() {
    return {
      'id': id ?? '',
      'is_crack': isCrack,
      'confidence': confidence,
      'area_ratio': areaRatio,
      'contour_length': contourLength,
      'processing_time_ms': processingTimeMs,
      'original_image_b64': originalImageB64,
      'overlay_image_b64': overlayImageB64,
      'intermediate_steps': intermediateSteps,
      'timestamp': timestamp.toUtc().toIso8601String(),
      'candidate_components': candidateComponents,
      'raw_candidate_components': rawCandidateComponents,
      'dominant_crack_override': dominantCrackOverride,
      'candidate_pixels': candidatePixels,
      'longest_candidate': longestCandidate,
      'mean_candidate_strength': meanCandidateStrength,
      'detection_score': detectionScore,
      'threshold_used': thresholdUsed,
      'shell_texture_score': shellTextureScore,
      'shell_texture_uniformity': shellTextureUniformity,
      'texture_anomaly_ratio': textureAnomalyRatio,
      'texture_candidate_pixels': textureCandidatePixels,
      'thin_crack_score': thinCrackScore,
      'thin_crack_detected': thinCrackDetected,
      'egg_detected': eggDetected,
      'egg_score': eggScore,
      'egg_size': eggSize,
      'egg_size_confidence': eggSizeConfidence,
      'egg_area_ratio': eggAreaRatio,
      'egg_width_pixels': eggWidthPixels,
      'egg_length_pixels': eggLengthPixels,
      'egg_width_ratio': eggWidthRatio,
      'egg_length_ratio': eggLengthRatio,
      'egg_size_score': eggSizeScore,
      'egg_size_memberships': eggSizeMemberships,
      'crack_size': crackSize,
      'crack_size_confidence': crackSizeConfidence,
      'crack_mask_b64': crackMaskB64,
      'crack_locations': crackLocations,
      'detection_iterations': detectionIterations,
      'search_iterations': searchIterations,
      'termination_reason': terminationReason,
      'sample_count': sampleCount,
      'crack_votes': crackVotes,
      'no_crack_votes': noCrackVotes,
      'decision_consistency': decisionConsistency,
      'area_consistent': areaConsistent,
      'area_consistency': areaConsistency,
      'area_mean_ratio': areaMeanRatio,
      'area_spread_ratio': areaSpreadRatio,
      'area_samples': areaSamples,
    };
  }

  static double _double(Object? value, {double fallback = 0}) {
    return value is num ? value.toDouble() : fallback;
  }

  @override
  List<Object?> get props => [
        id,
        isCrack,
        confidence,
        areaRatio,
        contourLength,
        processingTimeMs,
        timestamp,
        candidateComponents,
        rawCandidateComponents,
        dominantCrackOverride,
        candidatePixels,
        longestCandidate,
        meanCandidateStrength,
        detectionScore,
        thresholdUsed,
        shellTextureScore,
        shellTextureUniformity,
        textureAnomalyRatio,
        textureCandidatePixels,
        thinCrackScore,
        thinCrackDetected,
        eggDetected,
        eggScore,
        eggSize,
        eggSizeConfidence,
        eggAreaRatio,
        eggWidthPixels,
        eggLengthPixels,
        eggWidthRatio,
        eggLengthRatio,
        eggSizeScore,
        eggSizeMemberships,
        crackSize,
        crackSizeConfidence,
        crackMaskB64,
        crackLocations,
        detectionIterations,
        searchIterations,
        terminationReason,
        sampleCount,
        crackVotes,
        noCrackVotes,
        decisionConsistency,
        areaConsistent,
        areaConsistency,
        areaMeanRatio,
        areaSpreadRatio,
        areaSamples,
      ];
}
