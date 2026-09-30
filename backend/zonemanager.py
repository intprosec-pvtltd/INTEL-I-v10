import json
from pathlib import Path
import cv2
import numpy as np
from config import ZONE_CONFIG_PATH, DEFAULT_ZONE_MODE


class ZoneManager:
    def __init__(self, path=ZONE_CONFIG_PATH):
        self.path = Path(path)
        self.zones = {}
        self.load()

    def load(self):
        if self.path.exists():
            self.zones = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.zones = {}

    def save(self):
        self.path.write_text(json.dumps(self.zones, indent=2), encoding="utf-8")

    def set_zones(self, cam_id, zones):
        self.zones[cam_id] = zones
        self.save()

    def get_zones(self, cam_id):
        return self.zones.get(cam_id, [])

    def point_in_polygon(self, point, polygon):
        contour = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def matching_zones(self, cam_id, center):
        matched = []

        for zone in self.get_zones(cam_id):
            points = zone.get("points", [])

            if len(points) < 3:
                continue

            if self.point_in_polygon((float(center[0]), float(center[1])), points):
                matched.append(zone)

        return matched

    def validate(self, cam_id, rule, center):
        cam_zones = self.get_zones(cam_id)

        if not cam_zones:
            return {
                "allowed": DEFAULT_ZONE_MODE == "allow_all",
                "zone_name": "default",
                "zone_type": "default",
            }

        matched_zones = self.matching_zones(cam_id, center)

        if not matched_zones:
            return {
                "allowed": DEFAULT_ZONE_MODE == "allow_all",
                "zone_name": "outside_configured_zone",
                "zone_type": "outside",
            }

        for zone in matched_zones:
            allowed_rules = set(zone.get("rules", []))

            if not allowed_rules or rule in allowed_rules:
                return {
                    "allowed": True,
                    "zone_name": zone.get("name", "unnamed"),
                    "zone_type": zone.get("type", "normal"),
                }

        return {
            "allowed": False,
            "zone_name": matched_zones[0].get("name", "restricted"),
            "zone_type": matched_zones[0].get("type", "restricted"),
        }

    def is_restricted_zone(self, cam_id, center):
        for zone in self.matching_zones(cam_id, center):
            if zone.get("type") in {"restricted", "perimeter", "sensitive"}:
                return True, zone.get("name", "restricted")

        return False, None

    def draw(self, frame, cam_id):
        for zone in self.get_zones(cam_id):
            points = zone.get("points", [])

            if len(points) < 3:
                continue

            contour = np.array(points, dtype=np.int32)
            ztype = zone.get("type", "normal")

            color = (255, 200, 0)

            if ztype in {"restricted", "perimeter", "sensitive"}:
                color = (0, 0, 255)

            cv2.polylines(frame, [contour], True, color, 2)

            x, y = contour[0][0]

            cv2.putText(
                frame,
                zone.get("name", "zone"),
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return frame