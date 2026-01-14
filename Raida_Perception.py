#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raida_Perception.py

역할
- /scan (sensor_msgs/LaserScan)로부터 전방 ROI 포인트를 추출
- 콘(장애물) 포인트를 클러스터링하여 중심점을 추정
- 좌/우 콘을 분리하고 각각 2차 다항식으로 피팅
- /obs_lane_coeffs (Float32MultiArray)로 [La,Lb,Lc, Ra,Rb,Rc] 발행

계약(Decision.py와 호환)
- 좌표계(라이다/차량 기준):
    X: 전방(+), Y: 좌측(+)
- 발행 계수는 "Y = a*X^2 + b*X + c" (단위: meter)
  -> Decision의 LidarLane.get_error_norm()에서 그대로 사용 가능 fileciteturn12file4L141-L150

개선 포인트(원본을 그대로 복사하지 않음)
- Scan 좌표 변환/ROI 버그 방지 (전방 180도만 사용)
- Grid-bucket 기반 경량 DBSCAN(외부 라이브러리 없이)
- 좌/우 분리: 1차로 Y부호 기반 + 2차로 PCA 진행방향(EMA) 기반 정렬/정제
- 피팅 안정화: forward 정렬 후, 앞쪽/뒤쪽 outlier 제거, 최소점 개수 조건 강화
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


# -----------------------------
# Visualization (optional)
# -----------------------------
W, H = 640, 480
SCALE = 120  # 1m -> px
CENTER_X, CENTER_Y = W // 2, H - 60


@dataclass
class LidarPoint:
    x: float  # forward (m)
    y: float  # left (m)


class LidarConeLane:
    def __init__(self) -> None:
        rospy.init_node("raida_lane_prediction")

        gp = rospy.get_param

        # ROI
        self.roi_min = float(gp("~roi_min", 0.10))
        self.roi_max = float(gp("~roi_max", 2.00))
        self.fov_deg = float(gp("~fov_deg", 180.0))  # 전방 시야각

        # clustering
        self.use_dbscan = bool(gp("~use_dbscan", True))
        self.eps = float(gp("~cluster_eps", 0.18))          # DBSCAN eps (m)
        self.min_samples = int(gp("~min_samples", 3))       # DBSCAN min samples
        self.max_cluster_size = int(gp("~max_cluster_size", 60))

        # center extraction
        self.min_cluster_pts = int(gp("~min_cluster_pts", 3))

        # lane fit
        self.min_cones_per_side = int(gp("~min_cones_per_side", 4))
        self.fit_deg = int(gp("~fit_degree", 2))
        self.max_abs_y_m = float(gp("~max_abs_y_m", 2.0))

        # PCA EMA (주행 방향 벡터 안정화)
        self.pca_ema_alpha = float(gp("~pca_ema_alpha", 0.30))
        self._u_ema: Optional[np.ndarray] = None

        # Topics
        self.pub_coeffs = rospy.Publisher("/obs_lane_coeffs", Float32MultiArray, queue_size=1)

        # 다중 scan 토픽 대응 (launch 구성 차이 흡수)
        self._active_scan_topic: Optional[str] = None
        scan_topics = gp("~scan_topics", ["/scan", "/orda/scan", "/base_scan", "/orda/base_scan"])
        self._subs = []
        for t in scan_topics:
            self._subs.append(rospy.Subscriber(t, LaserScan, lambda m, _t=t: self._on_scan(m, _t), queue_size=1))

        # viz
        self.enable_viz = bool(gp("~enable_viz", False))
        self.viz_win = str(gp("~viz_window", "raida_lane_viz"))

        self._viz_points: List[LidarPoint] = []
        self._viz_centers: List[LidarPoint] = []
        self._viz_left: List[LidarPoint] = []
        self._viz_right: List[LidarPoint] = []
        self._viz_coeffs: List[float] = []

        rospy.loginfo("[Raida] ready | eps=%.2f min_samples=%d", self.eps, self.min_samples)

    # -----------------------------
    # Scan -> points
    # -----------------------------
    def _on_scan(self, msg: LaserScan, topic: str) -> None:
        if self._active_scan_topic is None:
            self._active_scan_topic = topic
            rospy.loginfo("[Raida] active scan topic: %s", topic)
        elif topic != self._active_scan_topic:
            return

        pts = self._extract_points(msg)
        self._viz_points = pts

        if not pts:
            self._publish([])
            self._viz_centers, self._viz_left, self._viz_right, self._viz_coeffs = [], [], [], []
            self._maybe_viz()
            return

        clusters = self._cluster_points(pts)
        centers = self._cluster_centers(clusters)
        self._viz_centers = centers

        left, right = self._split_left_right(centers)
        self._viz_left, self._viz_right = left, right

        coeffs = self._fit_lanes(left, right)
        self._viz_coeffs = coeffs

        self._publish(coeffs)
        self._maybe_viz()

    def _extract_points(self, msg: LaserScan) -> List[LidarPoint]:
        """LaserScan -> front ROI points (X forward, Y left)"""
        pts: List[LidarPoint] = []
        half_fov = math.radians(max(10.0, min(180.0, self.fov_deg)) * 0.5)

        ang = msg.angle_min
        for r in msg.ranges:
            if not math.isfinite(r):
                ang += msg.angle_increment
                continue
            if r < self.roi_min or r > self.roi_max:
                ang += msg.angle_increment
                continue

            # 전방 FOV만
            if ang < -half_fov or ang > half_fov:
                ang += msg.angle_increment
                continue

            # ROS LaserScan: angle=0 forward, +CCW
            x = float(r * math.cos(ang))
            y = float(r * math.sin(ang))

            # 전방만(조금이라도 뒤로 가는 포인트 제거)
            if x <= 0.02:
                ang += msg.angle_increment
                continue

            pts.append(LidarPoint(x=x, y=y))
            ang += msg.angle_increment

        # scan 각도 순서 유지(클러스터링 안정화)
        return pts

    # -----------------------------
    # Clustering
    # -----------------------------
    def _cluster_points(self, points: List[LidarPoint]) -> List[List[LidarPoint]]:
        if not points:
            return []
        if not self.use_dbscan:
            return self._seq_cluster(points, dist_thr=self.eps)

        P = np.array([[p.x, p.y] for p in points], dtype=float)
        labels = self._dbscan_grid(P, eps=self.eps, min_samples=self.min_samples)
        clusters: List[List[LidarPoint]] = []
        if labels.size == 0:
            return clusters

        max_label = int(labels.max()) if labels.max() >= 0 else -1
        for cid in range(max_label + 1):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            if idx.size > self.max_cluster_size:
                # 너무 큰 클러스터는 보통 벽/장애물/노이즈 덩어리일 가능성
                continue
            clusters.append([points[int(i)] for i in idx])

        return clusters

    def _seq_cluster(self, points: List[LidarPoint], dist_thr: float) -> List[List[LidarPoint]]:
        clusters: List[List[LidarPoint]] = []
        curr: List[LidarPoint] = [points[0]]
        for i in range(1, len(points)):
            dx = points[i].x - points[i - 1].x
            dy = points[i].y - points[i - 1].y
            d = math.hypot(dx, dy)
            if d <= dist_thr:
                curr.append(points[i])
            else:
                if len(curr) >= self.min_cluster_pts:
                    clusters.append(curr)
                curr = [points[i]]
        if len(curr) >= self.min_cluster_pts:
            clusters.append(curr)
        return clusters

    def _dbscan_grid(self, P: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """경량 DBSCAN. 반환 labels: -1(noise) or cluster_id 0.."""
        n = int(P.shape[0])
        if n == 0:
            return np.array([], dtype=np.int32)

        inv = 1.0 / float(eps)
        grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for i in range(n):
            cx = int(math.floor(P[i, 0] * inv))
            cy = int(math.floor(P[i, 1] * inv))
            grid[(cx, cy)].append(i)

        def neighbors(i: int) -> List[int]:
            cx = int(math.floor(P[i, 0] * inv))
            cy = int(math.floor(P[i, 1] * inv))
            out: List[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    out.extend(grid.get((cx + dx, cy + dy), []))
            # 실제 거리 필터
            if not out:
                return []
            pi = P[i]
            res = []
            for j in out:
                if j == i:
                    continue
                if np.hypot(P[j, 0] - pi[0], P[j, 1] - pi[1]) <= eps:
                    res.append(j)
            return res

        labels = np.full(n, -1, dtype=np.int32)
        visited = np.zeros(n, dtype=bool)
        cid = -1

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nbrs = neighbors(i)
            if len(nbrs) + 1 < min_samples:
                labels[i] = -1
                continue

            cid += 1
            labels[i] = cid
            seeds = list(nbrs)

            k = 0
            while k < len(seeds):
                j = seeds[k]
                if not visited[j]:
                    visited[j] = True
                    nbrs_j = neighbors(j)
                    if len(nbrs_j) + 1 >= min_samples:
                        # 확장
                        for q in nbrs_j:
                            if q not in seeds:
                                seeds.append(q)
                if labels[j] == -1:
                    labels[j] = cid
                k += 1

        return labels

    # -----------------------------
    # Cluster -> centers
    # -----------------------------
    def _cluster_centers(self, clusters: List[List[LidarPoint]]) -> List[LidarPoint]:
        centers: List[LidarPoint] = []
        for cl in clusters:
            if len(cl) < self.min_cluster_pts:
                continue
            arr = np.array([[p.x, p.y] for p in cl], dtype=float)
            cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            # 과도한 lateral outlier 제거
            if abs(cy) > self.max_abs_y_m:
                continue
            centers.append(LidarPoint(cx, cy))

        # 가까운(차 앞) 순 정렬
        centers.sort(key=lambda p: p.x)
        return centers

    # -----------------------------
    # Split left / right
    # -----------------------------
    def _split_left_right(self, centers: List[LidarPoint]) -> Tuple[List[LidarPoint], List[LidarPoint]]:
        if not centers:
            return [], []

        # 1) 1차 분리: Y 부호
        left = [p for p in centers if p.y >= 0.0]
        right = [p for p in centers if p.y < 0.0]

        # 양쪽이 극단적으로 쏠렸으면(예: Y 기준만으로 오분류) PCA로 보정
        if len(left) < 2 or len(right) < 2:
            left, right = self._split_by_pca(centers)

        # forward 정렬
        left.sort(key=lambda p: p.x)
        right.sort(key=lambda p: p.x)

        return left, right

    def _split_by_pca(self, centers: List[LidarPoint]) -> Tuple[List[LidarPoint], List[LidarPoint]]:
        """centers를 PCA 진행방향으로 정렬한 뒤, 진행방향에 직교하는 축 기준으로 좌/우 분리."""
        C = np.array([[p.x, p.y] for p in centers], dtype=float)
        if C.shape[0] < 3:
            # 최소 분리: Y 부호
            left = [p for p in centers if p.y >= 0.0]
            right = [p for p in centers if p.y < 0.0]
            return left, right

        mean = C.mean(axis=0, keepdims=True)
        X = C - mean
        cov = np.cov(X.T)
        w, v = np.linalg.eig(cov)
        u = v[:, int(np.argmax(w))].real
        u = u / (np.linalg.norm(u) + 1e-9)
        if u[0] < 0:
            u = -u

        # EMA로 진행방향 튐 감소
        if self._u_ema is None:
            self._u_ema = u
        else:
            a = float(self.pca_ema_alpha)
            u = (1 - a) * self._u_ema + a * u
            u = u / (np.linalg.norm(u) + 1e-9)
            self._u_ema = u

        # u에 직교하는 벡터
        n = np.array([-u[1], u[0]])

        proj = X @ n  # 직교축 투영값
        # 중앙(0)을 기준으로 부호로 분리
        left_idx = proj >= 0.0
        left = [centers[i] for i in range(len(centers)) if bool(left_idx[i])]
        right = [centers[i] for i in range(len(centers)) if not bool(left_idx[i])]

        return left, right

    # -----------------------------
    # Fit
    # -----------------------------
    def _fit_lanes(self, left: List[LidarPoint], right: List[LidarPoint]) -> List[float]:
        if len(left) < self.min_cones_per_side or len(right) < self.min_cones_per_side:
            return []

        lcoef = self._polyfit_xy(left)
        rcoef = self._polyfit_xy(right)

        if lcoef is None or rcoef is None:
            return []

        return [float(lcoef[0]), float(lcoef[1]), float(lcoef[2]),
                float(rcoef[0]), float(rcoef[1]), float(rcoef[2])]

    def _polyfit_xy(self, pts: List[LidarPoint]) -> Optional[np.ndarray]:
        """Y = f(X) 를 2차로 피팅 (meter)."""
        arr = np.array([[p.x, p.y] for p in pts], dtype=float)
        # forward 정렬 + outlier 완화(앞/뒤 1개씩 컷 가능)
        arr = arr[np.argsort(arr[:, 0])]
        if arr.shape[0] >= 6:
            arr = arr[1:-1]  # 극단값 2개 제거

        x = arr[:, 0]
        y = arr[:, 1]
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return None
        if arr.shape[0] < self.min_cones_per_side:
            return None

        deg = 2 if self.fit_deg >= 2 else 1
        try:
            coef = np.polyfit(x, y, deg)
            if deg == 1:
                coef = np.array([0.0, float(coef[0]), float(coef[1])], dtype=float)
            return coef.astype(float)
        except Exception:
            return None

    # -----------------------------
    # Publish / Viz
    # -----------------------------
    def _publish(self, coeffs: List[float]) -> None:
        m = Float32MultiArray()
        m.data = list(coeffs) if coeffs else []
        self.pub_coeffs.publish(m)

        rospy.loginfo_throttle(
            1.0,
            "[Raida] centers=%d | L=%d R=%d | coeffs=%s",
            len(self._viz_centers),
            len(self._viz_left),
            len(self._viz_right),
            "OK" if coeffs else "EMPTY",
        )

    def _maybe_viz(self) -> None:
        if not self.enable_viz:
            return

        img = np.zeros((H, W, 3), dtype=np.uint8)

        # axes
        cv2.line(img, (CENTER_X, 0), (CENTER_X, H), (60, 60, 60), 1)
        cv2.line(img, (0, CENTER_Y), (W, CENTER_Y), (60, 60, 60), 1)

        # raw points
        for p in self._viz_points:
            px = int(CENTER_X - p.y * SCALE)
            py = int(CENTER_Y - p.x * SCALE)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(img, (px, py), 1, (0, 180, 0), -1)

        # centers
        for p in self._viz_centers:
            px = int(CENTER_X - p.y * SCALE)
            py = int(CENTER_Y - p.x * SCALE)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(img, (px, py), 4, (255, 255, 255), 1)

        # left/right
        for p in self._viz_left:
            px = int(CENTER_X - p.y * SCALE)
            py = int(CENTER_Y - p.x * SCALE)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(img, (px, py), 6, (255, 0, 0), 2)
        for p in self._viz_right:
            px = int(CENTER_X - p.y * SCALE)
            py = int(CENTER_Y - p.x * SCALE)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(img, (px, py), 6, (0, 0, 255), 2)

        # fitted lanes (optional)
        if self._viz_coeffs and len(self._viz_coeffs) == 6:
            la, lb, lc, ra, rb, rc = self._viz_coeffs
            xs = np.arange(0.2, min(3.0, self.roi_max), 0.1)
            ptsL, ptsR, ptsC = [], [], []
            for x in xs:
                ly = la * x * x + lb * x + lc
                ry = ra * x * x + rb * x + rc
                cy = 0.5 * (ly + ry)
                ptsL.append((int(CENTER_X - ly * SCALE), int(CENTER_Y - x * SCALE)))
                ptsR.append((int(CENTER_X - ry * SCALE), int(CENTER_Y - x * SCALE)))
                ptsC.append((int(CENTER_X - cy * SCALE), int(CENTER_Y - x * SCALE)))
            if len(ptsL) > 1:
                cv2.polylines(img, [np.array(ptsL, np.int32)], False, (255, 100, 0), 2)
            if len(ptsR) > 1:
                cv2.polylines(img, [np.array(ptsR, np.int32)], False, (0, 100, 255), 2)
            if len(ptsC) > 1:
                cv2.polylines(img, [np.array(ptsC, np.int32)], False, (0, 255, 0), 2)

        cv2.putText(img, "Raida Lane Viz", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        cv2.imshow(self.viz_win, img)
        cv2.waitKey(1)

    def run(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    LidarConeLane().run()
