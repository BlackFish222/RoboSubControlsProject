import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from interfaces.msg import Yolov8Detection


@dataclass
class TrackState:
    track_id: int
    label: str
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float
    confidence: float
    persistence: int
    miss_count: int
    quality: float
    last_stamp_sec: float


class PerceptionFilterNode(Node):
    """
    Stabilizes YOLO detections with:
      - frame quality estimation
      - gating
      - persistence logic
      - simple prediction-based re-identification

    Publishes filtered detections using the same Yolov8Detection message type
    so tracker_node can subscribe without major changes.
    """

    def __init__(self) -> None:
        super().__init__("perception_filter_node")

        self.bridge = CvBridge()

        self.declare_parameter("image_topic", "video_frames_degraded")
        self.declare_parameter("detection_topic", "yolov8_detections")
        self.declare_parameter("output_topic", "filtered_detections")
        self.declare_parameter("quality_topic", "perception_quality")
        self.declare_parameter("degradation_status_topic", "degradation_status")

        self.declare_parameter("max_match_distance", 120.0)
        self.declare_parameter("min_confirm_frames", 3)
        self.declare_parameter("max_miss_frames", 10)
        self.declare_parameter("min_detector_confidence", 0.30)
        self.declare_parameter("min_quality_to_publish", 0.25)
        self.declare_parameter("prediction_gain", 0.7)

        image_topic = self.get_parameter("image_topic").value
        detection_topic = self.get_parameter("detection_topic").value
        output_topic = self.get_parameter("output_topic").value
        quality_topic = self.get_parameter("quality_topic").value
        degradation_status_topic = self.get_parameter("degradation_status_topic").value

        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        self.det_sub = self.create_subscription(
            Yolov8Detection, detection_topic, self.detection_callback, 50
        )
        self.degradation_sub = self.create_subscription(
            String, degradation_status_topic, self.degradation_callback, 50
        )

        self.filtered_pub = self.create_publisher(Yolov8Detection, output_topic, 50)
        self.quality_pub = self.create_publisher(String, quality_topic, 10)

        self.latest_frame: Optional[np.ndarray] = None
        self.current_blur_score: float = 0.0
        self.current_brightness: float = 0.0
        self.current_black_ratio: float = 0.0
        self.current_degradation_mode: str = "none"

        self.tracks: Dict[int, TrackState] = {}
        self.next_internal_id = 1

        self.get_logger().info(
            f"perception_filter_node started: {detection_topic} -> {output_topic}"
        )

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        self.latest_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.current_blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.current_brightness = float(np.mean(gray))
        self.current_black_ratio = float(np.mean(gray < 10))

        self.publish_quality_summary()

    def degradation_callback(self, msg: String) -> None:
        self.current_degradation_mode = msg.data

    def detection_callback(self, msg: Yolov8Detection) -> None:
        det = self.extract_detection(msg)
        if det is None:
            return

        det_id, label, conf, x, y, width, height = det

        min_detector_conf = float(self.get_parameter("min_detector_confidence").value)
        if conf < min_detector_conf:
            return

        now = self.now_sec()

        matched_track_id = self.find_best_track(label, x, y, width, height)
        if matched_track_id is None:
            matched_track_id = self.allocate_track_id(det_id)

        track = self.tracks.get(matched_track_id)
        if track is None:
            self.tracks[matched_track_id] = TrackState(
                track_id=matched_track_id,
                label=label,
                x=x,
                y=y,
                vx=0.0,
                vy=0.0,
                width=width,
                height=height,
                confidence=conf,
                persistence=1,
                miss_count=0,
                quality=0.0,
                last_stamp_sec=now,
            )
            track = self.tracks[matched_track_id]
        else:
            dt = max(1e-3, now - track.last_stamp_sec)
            pred_x = track.x + track.vx * dt
            pred_y = track.y + track.vy * dt

            new_vx = (x - track.x) / dt
            new_vy = (y - track.y) / dt

            alpha = float(self.get_parameter("prediction_gain").value)
            track.x = alpha * x + (1.0 - alpha) * pred_x
            track.y = alpha * y + (1.0 - alpha) * pred_y
            track.vx = alpha * new_vx + (1.0 - alpha) * track.vx
            track.vy = alpha * new_vy + (1.0 - alpha) * track.vy
            track.width = width
            track.height = height
            track.confidence = conf
            track.persistence += 1
            track.miss_count = 0
            track.last_stamp_sec = now

        track.quality = self.compute_track_quality(track)

        min_confirm_frames = int(self.get_parameter("min_confirm_frames").value)
        min_quality_to_publish = float(self.get_parameter("min_quality_to_publish").value)

        if track.persistence >= min_confirm_frames and track.quality >= min_quality_to_publish:
            out_msg = self.build_filtered_detection(msg, track)
            self.filtered_pub.publish(out_msg)

        self.age_unmatched_tracks(now, matched_track_id)

    def age_unmatched_tracks(self, now: float, matched_track_id: Optional[int]) -> None:
        max_miss_frames = int(self.get_parameter("max_miss_frames").value)
        stale_ids = []

        for track_id, track in self.tracks.items():
            if track_id == matched_track_id:
                continue

            track.miss_count += 1
            track.last_stamp_sec = now

            if track.miss_count > max_miss_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracks[track_id]

    def compute_track_quality(self, track: TrackState) -> float:
        conf_score = np.clip(track.confidence, 0.0, 1.0)

        blur_norm = np.clip(self.current_blur_score / 300.0, 0.0, 1.0)
        blackout_penalty = np.clip(self.current_black_ratio, 0.0, 1.0)

        persistence_norm = np.clip(track.persistence / 10.0, 0.0, 1.0)

        predicted_x = track.x + track.vx * 0.033
        predicted_y = track.y + track.vy * 0.033
        jump = math.hypot(track.x - predicted_x, track.y - predicted_y)
        jump_penalty = np.clip(jump / 200.0, 0.0, 1.0)

        mode_penalty = 0.0
        if self.current_degradation_mode in ("blackout",):
            mode_penalty = 0.6
        elif self.current_degradation_mode in ("haze", "gaussian_blur", "backscatter", "occlusion", "pixel_dropout"):
            mode_penalty = 0.2

        quality = (
            0.40 * conf_score
            + 0.20 * blur_norm
            + 0.25 * persistence_norm
            - 0.10 * jump_penalty
            - 0.20 * blackout_penalty
            - mode_penalty
        )

        return float(np.clip(quality, 0.0, 1.0))

    def find_best_track(
        self,
        label: str,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Optional[int]:
        best_track_id = None
        best_score = -1.0
        max_match_distance = float(self.get_parameter("max_match_distance").value)

        det_box = self.xywh_to_xyxy(x, y, width, height)

        for track_id, track in self.tracks.items():
            if track.label != label:
                continue

            dist = math.hypot(x - track.x, y - track.y)
            if dist > max_match_distance:
                continue

            track_box = self.xywh_to_xyxy(track.x, track.y, track.width, track.height)
            iou_score = self.iou(det_box, track_box)

            score = 0.6 * iou_score + 0.4 * (1.0 - min(dist / max_match_distance, 1.0))

            if score > best_score:
                best_score = score
                best_track_id = track_id

        return best_track_id

    def publish_quality_summary(self) -> None:
        msg = String()
        msg.data = (
            f"mode={self.current_degradation_mode}, "
            f"blur={self.current_blur_score:.3f}, "
            f"brightness={self.current_brightness:.3f}, "
            f"black_ratio={self.current_black_ratio:.3f}, "
            f"active_tracks={len(self.tracks)}"
        )
        self.quality_pub.publish(msg)

    def build_filtered_detection(
        self, original_msg: Yolov8Detection, track: TrackState
    ) -> Yolov8Detection:
        out = Yolov8Detection()

        # Adjust these field names if your msg differs.
        out.class_name = track.label
        out.tracking_id = track.track_id
        out.confidence = float(track.confidence)
        out.x = float(track.x)
        out.y = float(track.y)
        out.width = float(track.width)
        out.height = float(track.height)

        return out

    def extract_detection(
        self, msg: Yolov8Detection
    ) -> Optional[Tuple[int, str, float, float, float, float, float]]:
        """
        Adjust this method to your actual Yolov8Detection msg fields.
        Current assumptions:
          - class_name
          - tracking_id
          - confidence
          - x, y, width, height
        """
        try:
            det_id = int(msg.tracking_id)
            label = str(msg.class_name)
            conf = float(msg.confidence)
            x = float(msg.x)
            y = float(msg.y)
            width = float(msg.width)
            height = float(msg.height)
            return det_id, label, conf, x, y, width, height
        except AttributeError as exc:
            self.get_logger().error(f"Adjust extract_detection() to your message fields: {exc}")
            return None

    def allocate_track_id(self, detector_track_id: int) -> int:
        if detector_track_id >= 0:
            return detector_track_id

        new_id = self.next_internal_id
        self.next_internal_id += 1
        return new_id

    @staticmethod
    def xywh_to_xyxy(x: float, y: float, w: float, h: float):
        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x + 0.5 * w
        y2 = y + 0.5 * h
        return x1, y1, x2, y2

    @staticmethod
    def iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        if union <= 0.0:
            return 0.0
        return inter / union

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()