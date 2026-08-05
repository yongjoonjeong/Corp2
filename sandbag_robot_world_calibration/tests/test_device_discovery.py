from robot_calibration.device_discovery import VideoDeviceCandidate, is_c270_candidate


def candidate(**overrides):
    values = {
        "stable_path": "/dev/v4l/by-path/test-video-index0",
        "resolved_device": "/dev/video2",
        "product": "C270 HD WEBCAM",
        "serial": "",
        "usb_path": "pci-0000:00:14.0-usb-0:1:1.0",
        "vendor_id": "046d",
        "model_id": "0825",
    }
    values.update(overrides)
    return VideoDeviceCandidate(**values)


def test_c270_detected_by_product_name():
    assert is_c270_candidate(candidate())


def test_c270_detected_by_logitech_vid_pid():
    item = candidate(product="USB Camera", model_id="0826")
    assert is_c270_candidate(item)


def test_realsense_ir_is_not_c270():
    item = candidate(
        product="Intel RealSense D435 Depth",
        vendor_id="8086",
        model_id="0b07",
    )
    assert not is_c270_candidate(item)
