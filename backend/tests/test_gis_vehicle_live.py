import numpy as np

from services.gis_engine import project_pixel_to_geo


def test_project_pixel_to_geo_identity():
    result = project_pixel_to_geo(
        12.5,
        34.5,
        np.eye(3).tolist(),
    )
    assert result == (34.5, 12.5)


def test_project_pixel_to_geo_rejects_invalid_matrix():
    assert project_pixel_to_geo(10, 10, [[1, 0], [0, 1]]) is None


def test_project_pixel_to_geo_rejects_invalid_coordinate():
    assert project_pixel_to_geo(float("nan"), 10, np.eye(3).tolist()) is None
