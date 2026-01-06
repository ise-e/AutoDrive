#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple, Optional

import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """
    쿼터니언 -> yaw(평면 회전)만 추출
    """
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RaidaPerception:
    """
    LiDAR 기반 간단 PF 회피 노드
    - LaserScan -> base_link 포인트 변환
    - Potential Field로 방향(dir_deg) / 속도(vel) 생성
    - ROI 내 점들을 x,y,r 리스트로 obstacles 토픽에 발행
    """

    def __init__(self) -> None:
        rospy.init_node("raida_perception")

        self._load_params()

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(float(self.tf_cache_sec)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.dir_pub = rospy.Publisher(self.out_dir_topic, Float32, queue_size=1)
        self.vel_pub = rospy.Publisher(self.out_vel_topic, Float32, queue_size=1)
        self.obs_pub = rospy.Publisher(self.out_obs_topic, Float32MultiArray, queue_size=1)

        # Subscriber
        rospy.Subscriber(self.scan_topic, LaserScan, self._on_scan, queue_size=1)

        # 방향 EMA(벡터)
        self._pf_vec_ema = np.array([1.0, 0.0], dtype=np.float32)

        rospy.loginfo(
            "[Raida] ready | scan=%s base=%s | out: dir=%s vel=%s obs=%s",
            self.scan_topic, self.base_frame,
            self.out_dir_topic, self.out_vel_topic, self.out_obs_topic
        )

    # ------------------------
    # Params
    # ------------------------
    def _load_params(self) -> None:
        p = rospy.get_param
        defaults = {
            # IO
            "scan_topic": "/orda/sensors/lidar/scan",
            "base_frame": "base_link",
            "out_dir_topic": "/tunnel/direction",
            "out_vel_topic": "/tunnel/velocity",
            "out_obs_topic": "/orda/perc/obstacles",

            # TF
            "tf_timeout": 0.05,
            "tf_cache_sec": 0.5,

            # Scan preprocessing
            "max_range": 8.0,
            "robot_radius": 0.15,   # 로봇 외곽 기준으로 거리 보정

            # ROI (전방만 사용)
            "roi_x_max": 2.5,
            "roi_y_abs": 1.0,

            # obstacles publish (디버그/시각화용)
            "sample_max_points": 25,
            "sample_radius": 0.20,

            # PF common
            "pf_use_roi_for_pf": True,
            "forward_bias": 0.35,
            "left_bias": 0.0,

            # turn-based slow
            "turn_angle_for_full_slow_deg": 35.0,
            "turn_slow_k": 0.55,

            # velocity bounds (0~100 스케일을 가정)
            "vel_base": 100.0,
            "vel_min": 90.0,

            # (구버전 호환) eps
            "inv_eps": 1e-3,

            # ✅ 개선 PF 파라미터
            "pf_repulse_power": 2.0,      # repulsive weight: 1 / r^p
            "pf_repulse_eps": 1e-3,       # 0 나눔/폭주 방지
            "pf_max_points": 80,          # 가까운 점 상위 K개만 사용

            # ✅ 거리 기반 감속
            "pf_slow_dist_start_m": 1.2,  # 이 거리부터 감속 시작
            "pf_stop_dist_m": 0.35,       # 이 거리 이하면 vel_min 근처

            # ✅ 방향 EMA (0이면 비활성)
            "pf_dir_ema_alpha": 0.35,
        }

        for k, d in defaults.items():
            setattr(self, k, p("~" + k, d))

        # pf_repulse_eps 미설정 시 inv_eps를 따라가게
        if getattr(self, "pf_repulse_eps", None) is None:
            self.pf_repulse_eps = float(self.inv_eps)

    # ------------------------
    # TF / Geometry
    # ------------------------
    def _lookup_base_T_src(self, src_frame: str, stamp: rospy.Time) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        base_frame <- src_frame 변환을 (R,t)로 반환
        """
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame,           # target
                src_frame,                 # source
                stamp,
                timeout=rospy.Duration(float(self.tf_timeout)),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            rospy.logwarn_throttle(1.0, "[Raida] TF lookup failed (%s->%s)", src_frame, self.base_frame)
            return None

        tr = tfm.transform.translation
        qr = tfm.transform.rotation
        yaw = _yaw_from_quat(qr.x, qr.y, qr.z, qr.w)

        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        t = np.array([tr.x, tr.y], dtype=np.float32)
        return R, t

    # ------------------------
    # Scan -> Points
    # ------------------------
    def _scan_to_points_scanframe(self, scan: LaserScan) -> np.ndarray:
        """
        LaserScan -> (x,y) points in scan frame
        - 유효 range만 필터
        - robot_radius 만큼 보정
        """
        if not scan.ranges:
            return np.zeros((0, 2), dtype=np.float32)

        ranges = np.asarray(scan.ranges, dtype=np.float32)

        # range_max는 센서값/파라미터 중 더 작은 쪽 사용
        cap_max = float(min(float(scan.range_max), float(self.max_range)))
        rmin = float(scan.range_min)

        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= cap_max)
        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.float32)

        idx = np.nonzero(valid)[0].astype(np.float32)

        ang = float(scan.angle_min) + idx * float(scan.angle_increment)
        r = ranges[valid].astype(np.float32)

        # 로봇 반경만큼 빼서 “로봇 외곽 기준 거리”로 맞춤
        r = r - float(self.robot_radius)
        r = np.maximum(r, float(self.pf_repulse_eps))

        x = r * np.cos(ang)
        y = r * np.sin(ang)
        return np.stack([x, y], axis=1).astype(np.float32)

    def _transform_points(self, pts_scan: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        scan frame points -> base frame points
        """
        if pts_scan.size == 0:
            return pts_scan
        return (pts_scan @ R.T) + t

    # ------------------------
    # Potential Field (Improved)
    # ------------------------
    def _compute_pf(self, pts_base: np.ndarray) -> Tuple[float, float]:
        """
        pts_base: Nx2, base_link 기준 (x forward +, y left +)
        return: (dir_deg 0~360, vel 0~100)
        """
        if pts_base is None or pts_base.size == 0:
            # 장애물 없음: 전진 바이어스만
            dir_deg = (math.degrees(math.atan2(float(self.left_bias), float(self.forward_bias))) + 360.0) % 360.0
            return float(dir_deg), float(self.vel_base)

        xb = pts_base[:, 0].astype(np.float32)
        yb = pts_base[:, 1].astype(np.float32)

        # ROI로 PF 대상 제한 (옵션)
        if bool(self.pf_use_roi_for_pf):
            roi = (xb >= 0.0) & (xb <= float(self.roi_x_max)) & (np.abs(yb) <= float(self.roi_y_abs))
            if np.any(roi):
                xb = xb[roi]
                yb = yb[roi]

        if xb.size == 0:
            dir_deg = (math.degrees(math.atan2(float(self.left_bias), float(self.forward_bias))) + 360.0) % 360.0
            return float(dir_deg), float(self.vel_base)

        # 가까운 점 상위 K개만 사용 (회피 안정화)
        r2 = xb * xb + yb * yb
        K = int(max(5, int(self.pf_max_points)))
        if xb.size > K:
            sel = np.argpartition(r2, K)[:K]
            xb, yb, r2 = xb[sel], yb[sel], r2[sel]

        eps = float(self.pf_repulse_eps if self.pf_repulse_eps is not None else self.inv_eps)
        pwr = float(self.pf_repulse_power)

        # Repulsive force: (-x,-y) / r^p
        r = np.sqrt(np.maximum(r2, eps))
        w = 1.0 / np.power(np.maximum(r, eps), pwr)

        fx = float(np.sum((-xb) * w))
        fy = float(np.sum((-yb) * w))

        norm = math.hypot(fx, fy)
        if norm > 1e-6:
            ux, uy = fx / norm, fy / norm
        else:
            ux, uy = 0.0, 0.0

        # 전진/좌우 바이어스(“앞으로 가려는 성향”)
        vx = float(ux + float(self.forward_bias))
        vy = float(uy + float(self.left_bias))

        # 방향 EMA (벡터로 누적)
        a = float(self.pf_dir_ema_alpha)
        if a > 1e-6:
            v = np.array([vx, vy], dtype=np.float32)
            vn = float(np.linalg.norm(v))
            if vn > 1e-6:
                v /= vn
                self._pf_vec_ema = (1.0 - a) * self._pf_vec_ema + a * v

            ve = self._pf_vec_ema
            vx, vy = float(ve[0]), float(ve[1])

        dir_deg = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
        signed = dir_deg if dir_deg <= 180.0 else (dir_deg - 360.0)

        # 속도 1) 회전 기반 감속
        turn_ratio = min(abs(signed) / max(1e-6, float(self.turn_angle_for_full_slow_deg)), 1.0)
        vel_turn = float(self.vel_base) * (1.0 - float(self.turn_slow_k) * turn_ratio)

        # 속도 2) 최소거리 기반 감속
        min_dist = float(np.sqrt(float(np.min(r2))))
        slow_start = float(self.pf_slow_dist_start_m)
        stop_dist = float(self.pf_stop_dist_m)

        if slow_start <= stop_dist + 1e-6:
            vel_dist = float(self.vel_min)
        else:
            # stop_dist 이하 -> 0, slow_start 이상 -> 1
            t = (min_dist - stop_dist) / (slow_start - stop_dist)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            vel_dist = float(self.vel_min) + t * (float(self.vel_base) - float(self.vel_min))

        vel = min(vel_turn, vel_dist)
        vel = max(float(self.vel_min), min(float(self.vel_base), vel))
        return float(dir_deg), float(vel)

    # ------------------------
    # obstacles topic
    # ------------------------
    def _publish_obstacles(self, pts_base: np.ndarray) -> None:
        """
        ROI 내 점들을 (x,y,r) 반복 리스트로 발행
        """
        if pts_base is None or pts_base.size == 0:
            self.obs_pub.publish(Float32MultiArray(data=[]))
            return

        xb = pts_base[:, 0]
        yb = pts_base[:, 1]
        roi = (xb >= 0.0) & (xb <= float(self.roi_x_max)) & (np.abs(yb) <= float(self.roi_y_abs))
        if not np.any(roi):
            self.obs_pub.publish(Float32MultiArray(data=[]))
            return

        roi_pts = pts_base[roi]

        # 너무 많으면 균등 샘플링(시각화용)
        maxn = int(self.sample_max_points)
        if roi_pts.shape[0] > maxn:
            sel = np.linspace(0, roi_pts.shape[0] - 1, maxn).astype(np.int32)
            roi_pts = roi_pts[sel]

        rad = float(self.sample_radius)
        out = np.column_stack(
            [roi_pts, np.full((roi_pts.shape[0], 1), rad, dtype=np.float32)]
        ).reshape(-1).tolist()

        self.obs_pub.publish(Float32MultiArray(data=out))

    # ------------------------
    # ROS callback
    # ------------------------
    def _on_scan(self, scan: LaserScan) -> None:
        stamp = scan.header.stamp if scan.header.stamp else rospy.Time.now()
        src_frame = scan.header.frame_id if scan.header.frame_id else "laser"

        tf_rt = self._lookup_base_T_src(src_frame, stamp)
        if tf_rt is None:
            return
        R, t = tf_rt

        pts_scan = self._scan_to_points_scanframe(scan)
        if pts_scan.size == 0:
            # 장애물 없음 → 기본 전진
            dir_deg, vel = self._compute_pf(np.zeros((0, 2), dtype=np.float32))
            self.dir_pub.publish(Float32(data=float(dir_deg)))
            self.vel_pub.publish(Float32(data=float(max(0.0, min(100.0, vel)))))
            self.obs_pub.publish(Float32MultiArray(data=[]))
            return

        pts_base = self._transform_points(pts_scan, R, t)

        # PF 계산/발행
        dir_deg, vel = self._compute_pf(pts_base)
        self.dir_pub.publish(Float32(data=float(dir_deg)))
        self.vel_pub.publish(Float32(data=float(max(0.0, min(100.0, vel)))))

        # obstacles 발행
        self._publish_obstacles(pts_base)

    def run(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    RaidaPerception().run()
