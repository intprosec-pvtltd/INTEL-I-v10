from camera_integrations.sentinel_adapter import SentinelAdapter


class GenericVMSAdapter(SentinelAdapter):
    """Uses the existing catalogue connector while preserving one adapter contract."""
