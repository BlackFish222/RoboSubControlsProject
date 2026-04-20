import random
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class VisionDegradationNode(Node):
    """
    Degrades incoming camera images and republishes them.

    Pipeline:
        video_frames -> vision_degradation_node -> video_frames_degraded
    """

    def __init__(self) -> None:
        super().__init__("vision_degradation_node")

        self.bridge = CvBridge()

        self.declare_parameter("input_topic", "video_frames")
        self.declare_parameter("output_topic", "video_frames_degraded")
        self.declare_parameter("status_topic", "degradation_status")
        self.declare_parameter("blackout_topic", "blackout_active")

        self.declare_parameter("mode", "none")
        self.declare_parameter("severity", 0.0)
        self.declare_parameter("dropout_probability", 0.02)
        self.declare_parameter("blackout_type", "black_frame")  # black_frame | freeze_frame
        self.declare_parameter("occlusion_fraction", 0.25)
        self.declare_parameter("randomize_occlusion_position", True)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        status_topic = self.get_parameter("status_topic").value
        blackout_topic = self.get_parameter("blackout_topic").value

        self.image_sub = self.create_subscription(
            Image, input_topic, self.image_callback, 10
        )
        self.image_pub = self.create_publisher(Image, output_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.blackout_pub = self.create_publisher(Bool, blackout_topic, 10)

        self.last_output_frame: Optional[np.ndarray] = None

        self.get_logger().info(
            f"vision_degradation_node started: {input_topic} -> {output_topic}"
        )

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        mode = str(self.get_parameter("mode").value)
        severity = float(self.get_parameter("severity").value)
        severity = max(0.0, min(1.0, severity))

        degraded = self.apply_mode(frame, mode, severity)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(degraded, encoding="bgr8")
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)
        except CvBridgeError as exc:
            self.get_logger().error(f"Failed to publish degraded image: {exc}")
            return

        self.last_output_frame = degraded.copy()
        self.publish_status(mode)

    def publish_status(self, mode: str) -> None:
        status = String()
        status.data = mode
        self.status_pub.publish(status)

        blackout = Bool()
        blackout.data = (mode == "blackout")
        self.blackout_pub.publish(blackout)

    def apply_mode(self, frame: np.ndarray, mode: str, severity: float) -> np.ndarray:
        if mode == "none":
            return frame
        if mode == "gaussian_blur":
            return self.apply_gaussian_blur(frame, severity)
        if mode == "haze":
            return self.apply_haze(frame, severity)
        if mode == "color_shift":
            return self.apply_color_shift(frame, severity)
        if mode == "backscatter":
            return self.apply_backscatter(frame, severity)
        if mode == "pixel_dropout":
            return self.apply_pixel_dropout(frame, severity)
        if mode == "blackout":
            return self.apply_blackout(frame)
        if mode == "occlusion":
            return self.apply_occlusion(frame, severity)

        self.get_logger().warn(f"Unknown mode '{mode}', passing frame through.")
        return frame

    def apply_gaussian_blur(self, frame: np.ndarray, severity: float) -> np.ndarray:
        k = max(1, int(1 + severity * 20))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    def apply_haze(self, frame: np.ndarray, severity: float) -> np.ndarray:
        frame_f = frame.astype(np.float32) / 255.0
        transmission = max(0.2, 1.0 - severity)
        ambient = np.ones_like(frame_f) * 0.85
        hazy = frame_f * transmission + ambient * (1.0 - transmission)
        hazy = np.clip(hazy, 0.0, 1.0)
        hazy_u8 = (hazy * 255.0).astype(np.uint8)
        return self.apply_gaussian_blur(hazy_u8, severity * 0.4)

    def apply_color_shift(self, frame: np.ndarray, severity: float) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        hue_shift = random.uniform(-20.0, 20.0) * severity
        sat_scale = 1.0 + random.uniform(-0.5, 0.5) * severity
        val_scale = 1.0 + random.uniform(-0.5, 0.5) * severity

        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0.0, 255.0)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_backscatter(self, frame: np.ndarray, severity: float) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.float32)

        n_particles = int(100 + severity * 500)
        max_radius = max(2, int(2 + 8 * severity))

        for _ in range(n_particles):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.randint(1, max_radius)
            intensity = random.uniform(80.0, 255.0)
            cv2.circle(overlay, (x, y), radius, (intensity, intensity, intensity), -1)

        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=2.5)
        out = frame.astype(np.float32) + 0.25 * overlay
        out = np.clip(out, 0, 255).astype(np.uint8)
        return self.apply_haze(out, severity * 0.4)

    def apply_pixel_dropout(self, frame: np.ndarray, severity: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        base_p = float(self.get_parameter("dropout_probability").value)
        p = min(0.5, max(base_p, severity * 0.1))

        pixel_mask = np.random.rand(h, w) < p
        out[pixel_mask] = 0

        n_blocks = int(severity * 8)
        for _ in range(n_blocks):
            bw = random.randint(5, max(6, int(w * 0.1)))
            bh = random.randint(5, max(6, int(h * 0.1)))
            x1 = random.randint(0, max(0, w - bw))
            y1 = random.randint(0, max(0, h - bh))
            out[y1:y1 + bh, x1:x1 + bw] = 0

        return out

    def apply_blackout(self, frame: np.ndarray) -> np.ndarray:
        blackout_type = str(self.get_parameter("blackout_type").value)

        if blackout_type == "freeze_frame" and self.last_output_frame is not None:
            return self.last_output_frame.copy()

        return np.zeros_like(frame)

    def apply_occlusion(self, frame: np.ndarray, severity: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        configured_fraction = float(self.get_parameter("occlusion_fraction").value)
        frac = min(0.9, max(0.05, max(configured_fraction, severity)))

        occ_w = int(w * frac)
        occ_h = int(h * frac)

        randomize = bool(self.get_parameter("randomize_occlusion_position").value)
        if randomize:
            x1 = random.randint(0, max(0, w - occ_w))
            y1 = random.randint(0, max(0, h - occ_h))
        else:
            x1 = (w - occ_w) // 2
            y1 = (h - occ_h) // 2

        cv2.rectangle(out, (x1, y1), (x1 + occ_w, y1 + occ_h), (0, 0, 0), -1)
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionDegradationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()