"""Extensible camera-onboarding adapters."""

from camera_integrations.base_adapter import CameraIntegrationAdapter
from camera_integrations.generic_vms_adapter import GenericVMSAdapter
from camera_integrations.onvif_adapter import ONVIFAdapter
from camera_integrations.sentinel_adapter import SentinelAdapter

__all__ = ["CameraIntegrationAdapter", "GenericVMSAdapter", "ONVIFAdapter", "SentinelAdapter"]
