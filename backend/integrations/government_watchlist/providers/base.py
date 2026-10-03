from abc import ABC, abstractmethod
from typing import Any

class WatchlistProvider(ABC):
    @abstractmethod
    def fetch_records(self, source, credential: str | None) -> list[dict[str, Any]]:
        """Fetch external JSON records without mutating the remote system."""
