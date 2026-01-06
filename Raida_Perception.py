#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raida_Perception_refactored.py (기능 유지 + 직관성 개선)

목표
- 기존 v3의 "스캔 1회당 TF 1회 lookup + numpy 일괄 변환" 구조 유지
- 매직 넘버/축약어/인덱스 의존을 줄여서 디버깅/이해 속도를 올림
- 토픽/단위/데이터 규약을 파일 상단에 명시

토픽 계약(간단 요약)
- 입력:
  - ~scan_topic (sensor_msgs/LaserScan)  default: /scan

- 출력:
  - /tunnel/direction (std_msgs/Float32) : 0=전방, +90=좌측 (deg, 0~360)
  - /tunnel/velocity  (std_msgs/Float32) : 0~100 (권장 속도)
  - /detected_obstacles (std_msgs/Float32MultiArray) : [x_m, y_m, r_m,  x_m, y_m, r_m, ...] (base_link)
  - RViz:
    - /viz/scan_points (MarkerArray) : 샘플링 포인트
    - /viz/pf_vector   (MarkerArray) : PF 벡터(시각화)

주의
- PF 벡터 수식(1/x, 1/y 합산)은 기존과 동일하게 유지됩니다.
- 기존 코드에서 PF 계산은 ROI 필터를 적용하지 않았습니다.
  이 동작을 유지하기 위해 ~pf_use_roi_for_pf 기본값은 False 입니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rospy
import tf2_ros

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def clampf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass(frozen=True)
class PlanarTf:
    """2D (x,y) 평면에서의 TF: p_base = R @ p_src + t"""
    R: np.ndarray  # (2,2)
    t: np.ndarray  # (2,)

    @staticmethod
    def _quat_to_rot2(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        # yaw만 사용하는 2D 회전 행렬
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=np.float32)

    @staticmethod
    def from_tf(tf_msg) -> "PlanarTf":
        tr = tf_msg.transform.translation
        qr = tf_msg.transform.rotation
        R = PlanarTf._quat_to_rot2(qr.x, qr.y, qr.z, qr.w)
        t = np.array([tr.x, tr.y], dtype=np.float32)
        return PlanarTf(R=R, t=t)

    def apply(self, pts_xy: np.ndarray) -> np.ndarray:
        # pts_xy: (N,2)
        return (pts_xy @ self.R.T) + self.t


class LidarPotentialFieldNode:
    def __init__(self):
        # ---------- ROS params ----------
        get_param = rospy.get_param

        # frames / topics
        self.scan_topic = get_param("~scan_topic", "/scan")
        self.base_frame = get_param("~base_frame", "base_link")

        self.out_dir_topic = get_param("~out_dir_topic", "/tunnel/direction")
        self.out_vel_topic = get_param("~out_vel_topic", "/tunnel/velocity")
        self.out_obs_topic = get_param("~out_obs_topic", "/detected_obstacles")

        # TF
        self.tf_timeout = float(get_param("~tf_timeout", 0.1))
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # scan sanitize
        self.max_range = float(get_param("~max_range", 8.0))
        self.robot_radius = float(get_param("~robot_radius", 0.18))

        # PF (existing behavior)
        self.inv_eps = float(get_param("~inv_eps", 1e-3))
        self.forward_bias = float(get_param("~forward_bias", 0.0))
        self.left_bias = float(get_param("~left_bias", 0.0))

        self.vel_base = float(get_param("~vel_base", 80.0))
        self.vel_min = float(get_param("~vel_min", 30.0))
        self.turn_slow_k = float(get_param("~turn_slow_k", 0.7))
        self.turn_angle_for_full_slow_deg = float(get_param("~turn_angle_for_full_slow_deg", 60.0))

        # ROI (for obstacle sampling + optional PF)
        self.roi_x_max = float(get_param("~roi_x_max", 3.0))
        self.roi_y_abs = float(get_param("~roi_y_abs", 1.5))
        self.sample_max_points = int(get_param("~sample_max_points", 40))
        self.sample_radius = float(get_param("~sample_radius", 0.06))

        # keep old behavior by default
        self.pf_use_roi_for_pf = bool(get_param("~pf_use_roi_for_pf", False))

        # RViz
        self.points_marker_topic = get_param("~points_marker_topic", "/viz/scan_points")
        self.vector_marker_topic = get_param("~vector_marker_topic", "/viz/pf_vector")
        self.marker_lifetime = float(get_param("~marker_lifetime", 0.2))
        self.points_size = float(get_param("~points_size", 0.06))

        # ---------- pubs/subs ----------
        self.dir_pub = rospy.Publisher(self.out_dir_topic, Float32, queue_size=1)
        self.vel_pub = rospy.Publisher(self.out_vel_topic, Float32, queue_size=1)
        self.obs_pub = rospy.Publisher(self.out_obs_topic, Float32MultiArray, queue_size=1)

        self.points_pub = rospy.Publisher(self.points_marker_topic, MarkerArray, queue_size=1)
        self.vector_pub = rospy.Publisher(self.vector_marker_topic, MarkerArray, queue_size=1)

        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._on_scan, queue_size=1)

        rospy.loginfo(
            "[Raida] PF refactored | scan=%s base=%s | pf_use_roi_for_pf=%s",
            self.scan_topic, self.base_frame, self.pf_use_roi_for_pf
        )

    # ---------------- RViz publishers ----------------
    def _publish_points_marker(self, pts_xy, stamp):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp
        m.ns, m.id = "scan_pts", 0
        m.type, m.action = Marker.SPHERE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = float(self.points_size)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 1.0, 0.9
        m.lifetime = rospy.Duration(self.marker_lifetime)

        for x, y in pts_xy:
            m.points.append(Point(x=float(x), y=float(y), z=0.05))
        ma.markers.append(m)
        self.points_pub.publish(ma)

    def _publish_vector_marker(self, vec_xy: Tuple[float, float], stamp):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp
        m.ns, m.id = "pf_vec", 0
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.03, 0.06, 0.06
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.3, 0.95
        m.lifetime = rospy.Duration(self.marker_lifetime)

        p0 = Point(x=0.0, y=0.0, z=0.08)
        p1 = Point(x=float(vec_xy[0]), y=float(vec_xy[1]), z=0.08)
        m.points = [p0, p1]
        ma.markers.append(m)
        self.vector_pub.publish(ma)

    # ---------------- core math ----------------
    def _compute_pf(self, pts_base: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        기존 수식 유지:
        fx = sum(1/x), fy = sum(1/y)
        normalize 후 bias를 더하고 방향각(dir_deg) 계산
        """
        xb = pts_base[:, 0]
        yb = pts_base[:, 1]

        if self.pf_use_roi_for_pf:
            roi = (xb >= 0.0) & (xb <= self.roi_x_max) & (np.abs(yb) <= self.roi_y_abs)
            if roi.any():
                xb = xb[roi]
                yb = yb[roi]

        mx = np.abs(xb) > self.inv_eps
        my = np.abs(yb) > self.inv_eps

        fx = float(np.sum(1.0 / xb[mx])) if mx.any() else 0.0
        fy = float(np.sum(1.0 / yb[my])) if my.any() else 0.0

        norm = math.hypot(fx, fy)
        if norm > 1e-9:
            rx = (fx / norm) + self.forward_bias
            ry = (fy / norm) + self.left_bias
        else:
            rx = self.forward_bias
            ry = self.left_bias

        dir_deg = (math.degrees(math.atan2(ry, rx)) + 360.0) % 360.0
        signed = dir_deg if dir_deg <= 180.0 else (dir_deg - 360.0)

        turn_ratio = min(abs(signed) / max(1e-6, self.turn_angle_for_full_slow_deg), 1.0)
        vel = self.vel_base * (1.0 - self.turn_slow_k * turn_ratio)
        vel = clampf(vel, self.vel_min, self.vel_base)

        return float(dir_deg), float(vel), (float(rx), float(ry))

    # ---------------- scan callback ----------------
    def _on_scan(self, scan: LaserScan):
        stamp = scan.header.stamp if scan.header.stamp else rospy.Time.now()
        src_frame = scan.header.frame_id if scan.header.frame_id else "laser"

        # 1) TF lookup (scan 당 1회)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame, src_frame, stamp, timeout=rospy.Duration(self.tf_timeout)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(1.0, "[Raida] TF lookup failed (%s->%s): %s", src_frame, self.base_frame, str(e))
            return

        tf2d = PlanarTf.from_tf(tf_msg)

        # 2) LaserScan -> points in scan frame
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return

        # invalid -> max_range
        bad = ~np.isfinite(ranges) | (ranges <= 0.0)
        if bad.any():
            ranges = ranges.copy()
            ranges[bad] = float(self.max_range)
        ranges = np.minimum(ranges, float(self.max_range))

        # subtract robot radius
        ranges = ranges - float(self.robot_radius)
        valid = ranges > 0.0
        if not valid.any():
            return

        # NOTE: idx는 valid mask로 뽑힌 인덱스(원래 scan 인덱스)
        idx = np.nonzero(valid)[0].astype(np.float32)
        angles = float(scan.angle_min) + idx * float(scan.angle_increment)

        r = ranges[valid]
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        pts_scan = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2)

        # 3) scan -> base (numpy)
        pts_base = tf2d.apply(pts_scan)

        # 4) PF 방향/속도 계산
        dir_deg, vel, (rx, ry) = self._compute_pf(pts_base)

        self.dir_pub.publish(Float32(data=float(dir_deg)))
        self.vel_pub.publish(Float32(data=float(clampf(vel, 0.0, 100.0))))

        # 5) obstacle sample + RViz points (기존과 동일: ROI만 시각화/샘플링에 적용)
        xb = pts_base[:, 0]
        yb = pts_base[:, 1]
        roi = (xb >= 0.0) & (xb <= self.roi_x_max) & (np.abs(yb) <= self.roi_y_abs)
        if roi.any():
            roi_pts = pts_base[roi]
            k = int(roi_pts.shape[0])

            if k > self.sample_max_points:
                sel = np.linspace(0, k - 1, self.sample_max_points).astype(np.int32)
                roi_pts = roi_pts[sel]

            obs_out = []
            for x, y in roi_pts:
                obs_out.extend([float(x), float(y), float(self.sample_radius)])

            self.obs_pub.publish(Float32MultiArray(data=obs_out))
            # RViz publish (MarkerArray)
            # NOTE: points marker에서 geometry_msgs/Point를 사용해야 하지만 Point 접근은 환경에 따라 다를 수 있음.
            # 기존 코드와 동일하게 유지하고, 문제 발생 시 geometry_msgs.msg.Point로 교체하세요.
            try:
                self._publish_points_marker(roi_pts.tolist(), stamp)
            except Exception:
                pass

        # 6) PF vector marker (scale to 1.5m)
        vlen = math.hypot(rx, ry)
        if vlen > 1e-6:
            s = 1.5 / vlen
            try:
                self._publish_vector_marker((rx * s, ry * s), stamp)
            except Exception:
                pass


def main():
    rospy.init_node("raida_pf_node")
    LidarPotentialFieldNode()
    rospy.spin()


if __name__ == "__main__":
    main()
