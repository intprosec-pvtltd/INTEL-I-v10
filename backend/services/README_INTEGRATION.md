# INTEL-I Advanced Intelligence Add-on

This package implements the requested next-stage features as modular services.

## Included

1. Camera tamper detection
2. Frozen-frame detection
3. Cross-camera incident/entity correlation
4. Explainable risk scoring
5. Alert incident correlation
6. Evidence SHA-256 integrity
7. Explainable alerts
8. Operator feedback store
9. Adaptive FPS / workload policy
10. Bandwidth-aware policy
11. Person appearance features
12. Crowd intelligence
13. Audio anomaly gate
14. Character-level ANPR voting
15. Confidence calibration
16. EDSR x4 neural super-resolution
17. Generic ONNX hooks for dedicated low-light/dehaze models

## Important integration rule

Do not replace your existing `main.py` wholesale with a generated example. Import these
modules into the existing production `main.py` and wire them at the existing pipeline
boundaries.

## Recommended wiring

Camera/frame:
  cameraTamper.CameraTamperDetector
  frozenFrameDetector.FrozenFrameDetector
  cameraQuality.assess_frame_quality
  aiEnhancementModel.EDSR4x

Vehicle:
  personAppearance.appearance_features
  existing vehicleAttributes.infer_vehicle_attributes

Crowd:
  crowdIntelligence.CrowdIntelligence

ANPR:
  anprVoting.character_vote
  confidenceCalibration.ConfidenceCalibrator

Alerts:
  riskEngine.RiskEngine
  incidentCorrelation.IncidentCorrelator
  alertExplainability.build_alert_explanation
  evidenceIntegrity.build_evidence_manifest
  operatorFeedback.OperatorFeedbackStore

## Environment additions

FRAME_ENHANCEMENT_ENABLED=true
EDSR_MODEL_PATH=./models/EDSR_x4.pb
EDSR_ENABLED=true
EDSR_DEVICE=cpu
CAMERA_TAMPER_ENABLED=true
FROZEN_FRAME_ENABLED=true
RISK_SCORING_ENABLED=true
INCIDENT_CORRELATION_ENABLED=true
EVIDENCE_HASHING_ENABLED=true
OPERATOR_FEEDBACK_ENABLED=true
ADAPTIVE_PIPELINE_ENABLED=true
PERSON_APPEARANCE_ENABLED=true
CROWD_INTELLIGENCE_ENABLED=true
AUDIO_INTELLIGENCE_ENABLED=true
ANPR_CHARACTER_VOTING_ENABLED=true
CONFIDENCE_CALIBRATION_ENABLED=true

# Optional dedicated ONNX enhancement models.
LOW_LIGHT_ONNX_MODEL=./models/low_light_enhancement.onnx
DEHAZE_ONNX_MODEL=./models/dehaze.onnx
