"""Compatibility facade for the canonical INTEL-I vehicle correlation runtime.

Canonical stateful runtime: ``vehicleCorrelation.py``.
Policy-only deterministic helpers live in ``services.vehicleCorrelationPolicy``.
This module owns no correlation state.
"""
from services.vehicleCorrelationPolicy import evidence_fusion, VehiclePostGISGate, vehicle_postgis_gate


def _runtime():
    import vehicleCorrelation
    return vehicleCorrelation


def correlate_vehicle(*args, **kwargs): return _runtime().correlate_vehicle(*args, **kwargs)
def cleanup_global_vehicles(*args, **kwargs): return _runtime().cleanup_global_vehicles(*args, **kwargs)
def clear_camera_correlation_state(*args, **kwargs): return _runtime().clear_camera_correlation_state(*args, **kwargs)
def get_global_vehicle(*args, **kwargs): return _runtime().get_global_vehicle(*args, **kwargs)
def get_global_vehicle_id(*args, **kwargs): return _runtime().get_global_vehicle_id(*args, **kwargs)
def reassign_global_identity(*args, **kwargs): return _runtime().reassign_global_identity(*args, **kwargs)

__all__ = ["correlate_vehicle","cleanup_global_vehicles","clear_camera_correlation_state","get_global_vehicle","get_global_vehicle_id","reassign_global_identity","evidence_fusion","VehiclePostGISGate","vehicle_postgis_gate"]
