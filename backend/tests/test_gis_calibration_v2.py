import numpy as np
from services.gis_engine import project_pixel_to_geo


def test_v2_local_metric_projection():
    # Simple affine-like mapping in local metres: x_px -> x_m, y_px -> y_m.
    calibration = {
        "version": 2,
        "origin": {"latitude": 12.0, "longitude": 77.0},
        "homography_xy": [
            [0.1, 0.0, -48.0],
            [0.0, 0.1, -27.0],
            [0.0, 0.0, 1.0],
        ],
    }
    point = project_pixel_to_geo(500, 300, calibration=calibration)
    assert point is not None
    lat, lon = point
    assert abs(lat - (12.0 + 3.0 / 111320.0)) < 1e-9
    assert abs(lon - (77.0 + 2.0 / (111320.0 * np.cos(np.radians(12.0))))) < 1e-9
