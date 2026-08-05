from types import SimpleNamespace

import numpy as np

from three_camera_punch_feedback_node import colored_person_point_cloud_base


def test_colored_person_point_cloud_uses_mask_depth_and_base_transform():
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    color[2, 2] = (10, 20, 30)
    depth = np.zeros((4, 4), dtype=np.uint16)
    depth[2, 2] = 1000
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[2, 2] = 1.0
    intrinsics = SimpleNamespace(ppx=1.0, ppy=1.0, fx=2.0, fy=2.0)
    T_base_front = np.eye(4)
    T_base_front[:3, 3] = (100.0, 200.0, 300.0)

    points, colors = colored_person_point_cloud_base(
        color,
        depth,
        mask,
        intrinsics,
        0.001,
        T_base_front,
        stride=1,
    )

    np.testing.assert_allclose(points, ((600.0, 700.0, 1300.0),))
    np.testing.assert_array_equal(colors, ((10, 20, 30),))


def test_colored_person_point_cloud_rejects_background():
    color = np.full((2, 2, 3), 100, dtype=np.uint8)
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    mask = np.zeros((2, 2), dtype=np.float32)
    intrinsics = SimpleNamespace(ppx=0.0, ppy=0.0, fx=1.0, fy=1.0)

    points, colors = colored_person_point_cloud_base(
        color, depth, mask, intrinsics, 0.001, np.eye(4), stride=1
    )

    assert points.shape == (0, 3)
    assert colors.shape == (0, 3)
