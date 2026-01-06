#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray

def _yaw_from_quat(x, y, z, w):
    # yaw only
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

class RaidaPerception:
    def __init__(self):
        rospy.init_node("raida_perception")

        self._load_params()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(float(self.tf_cache_sec)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.dir_pub = rospy.Publisher(self.out_dir_topic, Float32, queue_size=1)
        self.vel_pub = rospy.Publisher(self.out_vel_topic, Float32, queue_size=1)
        self.obs_pub = rospy.Publisher(self.out_obs_topic, Float32MultiArray, queue_size=1)

        rospy.Subscriber(self.scan_topic, LaserScan, self._on_scan, queue_size=1)

    def _load_params(self):
        p = rospy.get_param
        defaults = {
            "scan_topic": "/orda/sensors/lidar/scan",
            "base_frame": "base_link",
            "tf_timeout": 0.05,
            "tf_cache_sec": 0.5,
            "max_range": 8.0,
            "robot_radius": 0.15,
            "roi_x_max": 2.5,
            "roi_y_abs": 1.0,
            "sample_max_points": 25,
            "sample_radius": 0.20,
            "inv_eps": 1e-3,
            "pf_use_roi_for_pf": True,
            "forward_bias": 0.35,
            "left_bias": 0.0,
            "turn_angle_for_full_slow_deg": 35.0,
            "turn_slow_k": 0.55,
            "vel_base": 100.0,
            "vel_min": 90.0,
            "out_dir_topic": "/tunnel/direction",
            "out_vel_topic": "/tunnel/velocity",
            "out_obs_topic": "/orda/perc/obstacles",
        }
        for k, d in defaults.items():
            setattr(self, k, p("~" + k, d))

    def _compute_pf(self, pts_base):
        xb, yb = pts_base[:, 0], pts_base[:, 1]
        if bool(self.pf_use_roi_for_pf):
            roi = (xb >= 0.0) & (xb <= float(self.roi_x_max)) & (np.abs(yb) <= float(self.roi_y_abs))
            if roi.any():
                xb, yb = xb[roi], yb[roi]

        eps = float(self.inv_eps)
        fx = float(np.sum(1.0 / xb[np.abs(xb) > eps])) if xb.size else 0.0
        fy = float(np.sum(1.0 / yb[np.abs(yb) > eps])) if yb.size else 0.0

        norm = math.hypot(fx, fy)
        if norm > 1e-6:
            rx, ry = (fx / norm) + float(self.forward_bias), (fy / norm) + float(self.left_bias)
        else:
            rx, ry = float(self.forward_bias), float(self.left_bias)

        dir_deg = (math.degrees(math.atan2(ry, rx)) + 360.0) % 360.0
        signed = dir_deg if dir_deg <= 180.0 else (dir_deg - 360.0)

        turn_ratio = min(abs(signed) / max(1e-6, float(self.turn_angle_for_full_slow_deg)), 1.0)
        vel = float(self.vel_base) * (1.0 - float(self.turn_slow_k) * turn_ratio)
        vel = max(float(self.vel_min), min(float(self.vel_base), vel))
        return float(dir_deg), float(vel)

    def _on_scan(self, scan):
        stamp = scan.header.stamp if scan.header.stamp else rospy.Time.now()
        src_frame = scan.header.frame_id if scan.header.frame_id else "laser"
        try:
            tfm = self.tf_buffer.lookup_transform(self.base_frame, src_frame, stamp, timeout=rospy.Duration(float(self.tf_timeout)))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            rospy.logwarn_throttle(1.0, "[Raida] TF lookup failed (%s->%s)", src_frame, self.base_frame)
            return

        tr = tfm.transform.translation
        qr = tfm.transform.rotation
        yaw = _yaw_from_quat(qr.x, qr.y, qr.z, qr.w)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        t = np.array([tr.x, tr.y], dtype=np.float32)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return
        bad = (~np.isfinite(ranges)) | (ranges <= 0.0)
        if bad.any():
            ranges = ranges.copy()
            ranges[bad] = float(self.max_range)
        ranges = np.minimum(ranges, float(self.max_range)) - float(self.robot_radius)
        valid = ranges > 0.0
        if not valid.any():
            return

        idx = np.nonzero(valid)[0].astype(np.float32)
        ang = float(scan.angle_min) + idx * float(scan.angle_increment)
        r = ranges[valid]
        pts_scan = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)
        pts_base = (pts_scan @ R.T) + t

        dir_deg, vel = self._compute_pf(pts_base)
        self.dir_pub.publish(Float32(data=float(dir_deg)))
        self.vel_pub.publish(Float32(data=float(max(0.0, min(100.0, vel)))))

        xb, yb = pts_base[:, 0], pts_base[:, 1]
        roi = (xb >= 0.0) & (xb <= float(self.roi_x_max)) & (np.abs(yb) <= float(self.roi_y_abs))
        if not roi.any():
            self.obs_pub.publish(Float32MultiArray(data=[]))
            return

        roi_pts = pts_base[roi]
        if roi_pts.shape[0] > int(self.sample_max_points):
            sel = np.linspace(0, roi_pts.shape[0] - 1, int(self.sample_max_points)).astype(np.int32)
            roi_pts = roi_pts[sel]

        rad = float(self.sample_radius)
        out = np.column_stack([roi_pts, np.full((roi_pts.shape[0], 1), rad, dtype=np.float32)]).reshape(-1).tolist()
        self.obs_pub.publish(Float32MultiArray(data=out))

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    RaidaPerception().run()
