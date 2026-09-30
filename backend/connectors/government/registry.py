from .base import GovernmentDataConnector
PROVIDERS = ("VAHAN", "SARATHI", "CCTNS", "AFIS", "NAFIS")
government_connectors = {name: GovernmentDataConnector(name) for name in PROVIDERS}
