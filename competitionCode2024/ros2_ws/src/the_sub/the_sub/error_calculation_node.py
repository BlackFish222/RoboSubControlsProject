import math
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
from std_msgs.msg import Empty, String

from interfaces.msg import ControlData, HeadingGoal


class ErrorCalculationNode(Node):
    """
    First-pass evaluator for degraded perception experiments.

    Current metrics:
      - heading RMSE
      - RMS control input
      - track lost count
      - dropout proxy
      - blackout recovery time
      - overshoot proxy
    """

    def __init__(self) -> None:
        super().__init__("error_calculation_node")

        self.declare_parameter("heading_goal_topic", "heading_goal")
        self.declare_parameter("control_data_topic", "control_data")
        self.declare_parameter("track_lost_topic", "track_lost")
        self.declare_parameter("degradation_status_topic", "degradation_status")
        self.declare_parameter("quality_topic", "perception_quality")
        self.declare_parameter("blackout_topic", "blackout_active")
        self.declare_parameter("metrics_topic", "live_metrics")
        self.declare_parameter("heading_recovery_threshold_deg", 5.0)

        heading_goal_topic = self.get_parameter("heading_goal_topic").value
        control_data_topic = self.get_parameter("control_data_topic").value
        track_lost_topic = self.get_parameter("track_lost_topic").value
        degradation_status_topic = self.get_parameter("degradation_status_topic").value
        quality_topic = self.get_parameter("quality_topic").value
        blackout_topic = self.get_parameter("blackout_topic").value
        metrics_topic = self.get_parameter("metrics_topic").value

        self.goal_sub = self.create_subscription(
            HeadingGoal, heading_goal_topic, self.heading_goal_callback, 50
        )
        self.control_sub = self.create_subscription(
            ControlData, control_data_topic, self.control_callback, 50
        )
        self.track_lost_sub = self.create_subscription(
            Empty, track_lost_topic, self.track_lost_callback, 50
        )
        self.degradation_sub = self.create_subscription(
            String, degradation_status_topic, self.degradation_callback, 50
        )
        self.quality_sub = self.create_subscription(
            String, quality_topic, self.quality_callback, 50
        )
        self.blackout_sub = self.create_subscription(
            String, blackout_topic, self.blackout_callback, 50
        )

        self.metrics_pub = self.create_publisher(String, metrics_topic, 10)

        self.current_goal_yaw: Optional[float] = None
        self.current_yaw: Optional[float] = None
        self.current_control_input: Optional[float] = None

        self.heading_errors: List[float] = []
        self.control_inputs: List[float] = []

        self.track_lost_count = 0
        self.dropout_count = 0
        self.max_overshoot = 0.0

        self.current_mode = "none"
        self.last_quality_summary = ""

        self.blackout_active = False
        self.blackout_end_time: Optional[float] = None
        self.recovery_times: List[float] = []

        self.recovery_threshold_rad = math.radians(
            float(self.get_parameter("heading_recovery_threshold_deg").value)
        )

        self.timer = self.create_timer(1.0, self.publish_metrics)

        self.get_logger().info("error_calculation_node started.")

    def heading_goal_callback(self, msg: HeadingGoal) -> None:
        # Adjust field if your HeadingGoal message differs
        self.current_goal_yaw = self.quaternion_to_yaw(msg.target_heading)

    def control_callback(self, msg: ControlData) -> None:
        # Adjust these field names if needed
        self.current_yaw = self.quaternion_to_yaw(msg.attitude)
        self.current_control_input = float(msg.turn)

        if self.current_goal_yaw is None or self.current_yaw is None:
            return

        error = self.angle_error(self.current_goal_yaw, self.current_yaw)
        self.heading_errors.append(error)

        if self.current_control_input is not None:
            self.control_inputs.append(self.current_control_input)

        self.update_overshoot()
        self.update_blackout_recovery(error)

    def track_lost_callback(self, _msg: Empty) -> None:
        self.track_lost_count += 1
        self.dropout_count += 1

    def degradation_callback(self, msg: String) -> None:
        self.current_mode = msg.data

    def quality_callback(self, msg: String) -> None:
        self.last_quality_summary = msg.data

    def blackout_callback(self, msg: String) -> None:
        # If you keep blackout_active as Bool, change subscriber type to Bool.
        active = msg.data.lower() == "true"

        now = self.now_sec()
        if active and not self.blackout_active:
            self.blackout_active = True
        elif not active and self.blackout_active:
            self.blackout_active = False
            self.blackout_end_time = now

    def update_overshoot(self) -> None:
        if self.current_goal_yaw is None or self.current_yaw is None:
            return

        goal_mag = abs(self.current_goal_yaw)
        if goal_mag < 1e-6:
            return

        overshoot = max(
            0.0,
            (abs(self.current_yaw) - abs(self.current_goal_yaw)) / goal_mag,
        )
        self.max_overshoot = max(self.max_overshoot, overshoot)

    def update_blackout_recovery(self, error: float) -> None:
        if self.blackout_active:
            return

        if self.blackout_end_time is None:
            return

        if abs(error) <= self.recovery_threshold_rad:
            recovery_time = self.now_sec() - self.blackout_end_time
            self.recovery_times.append(recovery_time)
            self.blackout_end_time = None

    def publish_metrics(self) -> None:
        heading_rmse = self.compute_rmse(self.heading_errors)
        rms_control = self.compute_rmse(self.control_inputs)
        mean_recovery = float(np.mean(self.recovery_times)) if self.recovery_times else 0.0

        msg = String()
        msg.data = (
            f"mode={self.current_mode}, "
            f"heading_rmse={heading_rmse:.6f}, "
            f"rms_control_input={rms_control:.6f}, "
            f"overshoot={self.max_overshoot:.6f}, "
            f"track_lost_count={self.track_lost_count}, "
            f"dropout_count={self.dropout_count}, "
            f"blackout_recovery_time={mean_recovery:.6f}, "
            f"quality=({self.last_quality_summary})"
        )
        self.metrics_pub.publish(msg)

    @staticmethod
    def compute_rmse(values: List[float]) -> float:
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(arr))))

    @staticmethod
    def angle_error(a: float, b: float) -> float:
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    @staticmethod
    def quaternion_to_yaw(q: Quaternion) -> float:
        x, y, z, w = q.x, q.y, q.z, q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ErrorCalculationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()