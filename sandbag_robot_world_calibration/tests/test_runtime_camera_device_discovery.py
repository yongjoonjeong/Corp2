import tempfile
import unittest
from pathlib import Path

from camera_device_discovery import (
    discover_c270_capture_devices,
    resolve_stereo_webcam_devices,
)


class CameraDeviceDiscoveryTests(unittest.TestCase):
    def make_video_node(
        self,
        sysfs: Path,
        dev: Path,
        usb: Path,
        video_name: str,
        product: str,
        index: int,
    ) -> None:
        entry = sysfs / video_name
        entry.mkdir(parents=True)
        (entry / "name").write_text(product, encoding="utf-8")
        (entry / "index").write_text(str(index), encoding="utf-8")
        usb.mkdir(parents=True, exist_ok=True)
        (entry / "device").symlink_to(usb, target_is_directory=True)
        (dev / video_name).touch()

    def test_discovers_only_c270_capture_nodes_in_physical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = root / "sys"
            dev = root / "dev"
            sysfs.mkdir()
            dev.mkdir()
            # Video numbering is intentionally opposite to physical USB order.
            self.make_video_node(
                sysfs, dev, root / "usb" / "3-2", "video4", "C270 HD WEBCAM", 0
            )
            self.make_video_node(
                sysfs, dev, root / "usb" / "3-1", "video12", "C270 HD WEBCAM", 0
            )
            self.make_video_node(
                sysfs, dev, root / "usb" / "3-1", "video13", "C270 HD WEBCAM", 1
            )
            self.make_video_node(
                sysfs, dev, root / "usb" / "6-1", "video8", "RealSense", 0
            )
            devices = discover_c270_capture_devices(sysfs, dev)
        self.assertEqual(devices, [str(dev / "video12"), str(dev / "video4")])

    def test_resolves_auto_pair_and_optional_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = root / "sys"
            dev = root / "dev"
            sysfs.mkdir()
            dev.mkdir()
            self.make_video_node(
                sysfs, dev, root / "usb" / "3-1", "video4", "C270", 0
            )
            self.make_video_node(
                sysfs, dev, root / "usb" / "5-1", "video6", "C270", 0
            )
            left, right, automatic = resolve_stereo_webcam_devices(
                "auto", "auto", True, sysfs, dev
            )
        self.assertTrue(automatic)
        self.assertEqual((left, right), (str(dev / "video6"), str(dev / "video4")))

    def test_rejects_mixed_auto_and_explicit_paths(self):
        with self.assertRaisesRegex(ValueError, "둘 다 auto"):
            resolve_stereo_webcam_devices("auto", "/dev/video6")

    def test_keeps_explicit_pair(self):
        self.assertEqual(
            resolve_stereo_webcam_devices("/dev/video20", "/dev/video22"),
            ("/dev/video20", "/dev/video22", False),
        )

if __name__ == "__main__":
    unittest.main()
