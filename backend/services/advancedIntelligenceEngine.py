from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from .cameraTamper import CameraTamperDetector
from .frozenFrameDetector import FrozenFrameDetector
from .riskEngine import RiskEngine
from .incidentCorrelation import IncidentCorrelator, Observation
from .alertExplainability import build_alert_explanation
from .evidenceIntegrity import build_evidence_manifest
from .operatorFeedback import OperatorFeedbackStore
from .adaptivePipeline import AdaptivePipelineController
from .personAppearance import appearance_features
from .crowdIntelligence import CrowdIntelligence
from .audioIntelligence import AudioIntelligence
from .anprVoting import OCRCandidate, character_vote
from .confidenceCalibration import ConfidenceCalibrator
from .aiEnhancementModel import EDSR4x
from .darkIR import DarkIRRestorer

class AdvancedIntelligenceEngine:
    """
    Per-application orchestrator.

    Keep this object in main.py as a singleton and call the small methods at
    the corresponding processing stages. Persistence remains the responsibility
    of the existing INTEL-I PostgreSQL/CRUD layer.
    """

    def __init__(
        self,
        edsr_model_path: str = "./models/EDSR_x4.pb",
        edsr_device: str = "cpu",
        darkir_model_path: str = "./models/DarkIR_384.pt",
        darkir_device: str = "auto",
        darkir_max_width: int = 1280,
        darkir_max_height: int = 720,
        darkir_strict: bool = False,
        darkir_sha256: str = "",
    ):
        self.tamper = {}
        self.frozen = {}
        self.crowd = {}
        self.audio = AudioIntelligence()
        self.risk = RiskEngine()
        self.incidents = IncidentCorrelator()
        self.feedback = OperatorFeedbackStore()
        self.adaptive = AdaptivePipelineController()
        self.calibrator = ConfidenceCalibrator()
        self.edsr = EDSR4x(edsr_model_path, edsr_device)
        self.darkir = DarkIRRestorer(
            darkir_model_path,
            darkir_device,
            max_width=darkir_max_width,
            max_height=darkir_max_height,
            strict=darkir_strict,
            expected_sha256=darkir_sha256,
        )

    def restore_low_light(
        self,
        image: np.ndarray,
        quality: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        quality = quality or {}
        mode = str(quality.get("mode") or "").strip().upper()
        flags = {str(v).strip().lower() for v in quality.get("flags", [])}
        low_light = force or mode == "LOW_LIGHT" or bool(
            {"low_light", "very_dark"} & flags
        )
        if not low_light:
            return {
                "image": image,
                "model_used": False,
                "reason": "not_low_light",
            }

        result = self.darkir.enhance(image)
        return {
            "image": result.get("frame", image),
            **result,
        }

    def darkir_status(self) -> Dict[str, Any]:
        return self.darkir.status()

    def camera_health(
        self,
        camera_id: str,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        tamper = self.tamper.setdefault(camera_id, CameraTamperDetector())
        frozen = self.frozen.setdefault(camera_id, FrozenFrameDetector())

        t = tamper.update(frame)
        f = frozen.update(frame)

        return {
            "camera_id": camera_id,
            "tamper": asdict(t),
            "frozen": asdict(f),
        }

    def reset_camera(self, camera_id: str) -> None:
        """Reset camera-local visual baselines after a verified discontinuity."""
        self.tamper.pop(str(camera_id), None)
        self.frozen.pop(str(camera_id), None)

    def adaptive_policy(
        self,
        source_fps: float,
        activity_score: float,
        quality_score: float,
        network_limited: bool = False,
    ) -> Dict[str, Any]:
        return asdict(
            self.adaptive.policy(
                source_fps=source_fps,
                activity_score=activity_score,
                quality_score=quality_score,
                network_limited=network_limited,
            )
        )

    def upscale_when_needed(
        self,
        image: np.ndarray,
        quality_score: float,
        force: bool = False,
    ) -> Dict[str, Any]:
        use = force or float(quality_score) < 0.60
        if not use:
            return {"image": image, "model_used": False}

        out = self.edsr.upscale(image)
        return {
            "image": out,
            "model_used": out is not image,
        }

    def person_appearance(self, person_crop: np.ndarray) -> Dict[str, Any]:
        return appearance_features(person_crop)

    def crowd_update(
        self,
        camera_id: str,
        count: int,
        frame_area: int,
        timestamp_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        detector = self.crowd.setdefault(camera_id, CrowdIntelligence())
        result = detector.update(
            count=count,
            frame_area=frame_area,
            now_seconds=time.time() if timestamp_seconds is None else timestamp_seconds,
        )
        return asdict(result)

    def audio_update(self, pcm: np.ndarray) -> Dict[str, Any]:
        return asdict(self.audio.analyze_pcm(pcm))

    def stable_plate(
        self,
        candidates: list[OCRCandidate],
        quality_score: float,
        temporal_support: float,
        raw_confidence: float,
    ) -> Dict[str, Any]:
        text, voting_score = character_vote(candidates)
        calibrated = self.calibrator.calibrate(
            raw_confidence,
            quality_score=quality_score,
            temporal_support=temporal_support,
        )
        return {
            "plate": text,
            "vote_score": voting_score,
            "calibrated_confidence": calibrated,
        }

    def create_risk(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return asdict(self.risk.from_payload(payload))

    def correlate(
        self,
        event_id: str,
        camera_id: str,
        entity_key: str,
        event_type: str,
        confidence: float,
        evidence: Optional[Dict[str, Any]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        event = Observation(
            event_id=event_id,
            camera_id=camera_id,
            timestamp=time.time() if timestamp is None else timestamp,
            entity_key=entity_key,
            event_type=event_type,
            confidence=confidence,
            latitude=latitude,
            longitude=longitude,
            evidence=evidence or {},
        )
        incident = self.incidents.ingest(event)

        return {
            "incident_id": incident.incident_id,
            "entity_key": incident.entity_key,
            "risk_score": incident.risk_score,
            "severity": incident.severity,
            "event_count": len(incident.events),
            "camera_count": len({e.camera_id for e in incident.events}),
        }

    def explain_alert(
        self,
        rule: str,
        camera_id: str,
        confidence: Optional[float],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_alert_explanation(
            rule=rule,
            camera_id=camera_id,
            confidence=confidence,
            evidence=evidence,
        )

    def evidence_manifest(
        self,
        alert_id: str,
        snapshot_path: Optional[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_evidence_manifest(
            alert_id=alert_id,
            snapshot_path=snapshot_path,
            metadata=metadata,
        )

    def operator_feedback(
        self,
        alert_id: str,
        decision: str,
        operator_id: str,
        note: str = "",
    ) -> Dict[str, Any]:
        record = self.feedback.add(
            alert_id,
            decision,
            operator_id,
            note,
        )
        return asdict(record)
