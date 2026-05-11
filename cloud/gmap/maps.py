"""Google Maps wrapper. Provides walking directions and the next step-by-step hint."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NavStep:
    distance_m: int
    duration_s: int
    instruction_zh: str
    end_lat: float
    end_lng: float


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


class GmapClient:
    def __init__(self, api_key: str, language: str = "zh-TW") -> None:
        if not api_key:
            raise ValueError("GMAP_API_KEY is empty")
        import googlemaps
        self._g = googlemaps.Client(api_key)
        self.language = language

    async def walk_directions(self, origin: str | tuple[float, float],
                              destination: str | tuple[float, float]) -> list[NavStep]:
        def _call():
            return self._g.directions(origin, destination,
                                      mode="walking",
                                      language=self.language,
                                      alternatives=False)
        raw = await asyncio.to_thread(_call)
        if not raw:
            return []
        legs = raw[0].get("legs", [])
        steps: list[NavStep] = []
        for leg in legs:
            for s in leg.get("steps", []):
                steps.append(NavStep(
                    distance_m=int(s.get("distance", {}).get("value", 0)),
                    duration_s=int(s.get("duration", {}).get("value", 0)),
                    instruction_zh=_strip_html(s.get("html_instructions", "")),
                    end_lat=float(s.get("end_location", {}).get("lat", 0.0)),
                    end_lng=float(s.get("end_location", {}).get("lng", 0.0)),
                ))
        return steps

    def pick_next(self, steps: list[NavStep],
                  current: Optional[tuple[float, float]] = None) -> Optional[NavStep]:
        if not steps:
            return None
        # Without live GPS we just return the first remaining step.
        # When you have IMU-derived displacement or a phone GPS feed,
        # advance past steps whose end_lat/lng is "passed".
        return steps[0]
