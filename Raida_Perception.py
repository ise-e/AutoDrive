#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple, Optional

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

class RaidaLanePrediction:
    """
    LiDAR 기반 간단 PF 회피 노드
    - LaserScan -> base_link 포인트 변환
    - Potential Field로 방향(dir_deg) / 속도(vel) 생성
    - ROI 내 점들을 x,y,r 리스트로 obstacles 토픽에 발행
    """

    def __init__(self) -> None:
        rospy.init_node("raida_lane_prediction")

        # Params
        self.distance_threshold = 0.1  # 같은 물체로 판단할 점들 사이의 최대 거리 (m)
        self.min_cluster_size = 3      # 유효한 물체로 볼 최소 점 개수
        self.max_cluster_size = 30     # 라바콘 크기를 고려한 최대 점 개수
        self.roi_front_limit = 3.0     # 전방 탐색 제한 거리 (m) 
        self.isdetected = False         # 라인 감지 확인용

        # 구독자 및 발행자 초기화
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # [La, Lb, Lc, Ra, Rb, Rc]로 발행
        self.pub_coeffs = rospy.Publisher("/obs_lane_coeffs", Float32MultiArray, queue_size=1)


    def fit_circle(self, points):
        """클러스터 중심점 계산"""
        x = points[:, 0]
        y = points[:, 1]

        A = np.column_stack((x, y, np.ones(len(x))))
        b = -(x**2 + y**2)

        # 식: x^2 + y^2 + Dx + Ey + F = 0
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = result
        
        center_x = -D / 2
        center_y = -E / 2

        dist = center_x**2 + center_y**2

        return center_x, center_y, dist

    # 라이다 데이터를 처리하여 클러스터링 수행
    def scan_callback(self, msg):
        points = []
        
        # 전처리: 극좌표를 직교좌표(x, y)로 변환
        for i, r in enumerate(msg.ranges):
            # 무한대나 유효하지 않은 값 제외
            if r < 0.05 or r > self.roi_front_limit:
                continue
        
            angle = msg.angle_min + i * msg.angle_increment

            # 전방 180도만 처리 (ROI 설정)
            if angle < -math.pi/2 or angle > math.pi/2:
                x = -r * math.cos(angle)
                y = -r * math.sin(angle)
                points.append([x, y])

        # --- [2] 유클리드 클러스터링 알고리즘 ---
        clusters = []
        if len(points) > 0:
            current_cluster = [points[0]]
            for i in range(1, len(points)):
                # 이전 점과의 거리 계산
                dist = math.sqrt((points[i][0] - points[i-1][0])**2 + 
                                (points[i][1] - points[i-1][1])**2)
                
                if dist < self.distance_threshold:
                    current_cluster.append(points[i])
                else:
                    if self.min_cluster_size <= len(current_cluster) <= self.max_cluster_size:
                        clusters.append(current_cluster)
                    current_cluster = [points[i]]
            # 마지막 클러스터 추가
            if self.min_cluster_size <= len(current_cluster) <= self.max_cluster_size:
                clusters.append(current_cluster)

        # 클러스터별 좌표 및 좌우 차선 분류
        obstacle_center_arr = []
        left_line = []
        right_line = []

        for cluster in clusters:
            arr = np.array(cluster)
            
            if len(arr) >= self.min_cluster_size:
                center = np.array(self.fit_circle(arr))
                obstacle_center_arr.append(center)

        obstacle_center_arr = sorted(obstacle_center_arr, key=lambda x: x[2])

        for point in obstacle_center_arr:
            if len(right_line) and len(left_line):
                dist_to_left = np.sum((left_line[-1]-point[:2])**2)
                dist_to_right = np.sum((right_line[-1]-point[:2])**2)

                if dist_to_left < dist_to_right:
                    left_line.append(point)
                else:
                    right_line.append(point)
            else:
                if point[1] > 0:
                    left_line.append(point)
                else:
                    right_line.append(point)

        # 포맷: [L_a, L_b, L_c,  R_a, R_b, R_c]
        if len(points) == 0 or len(right_line) < 3 or len(left_line) < 3: 
            # 라바콘이 없거나 둘중 한쪽 차선의 라바콘이 2개 이하일 때, 장애물 없다고 판단.
            coeffs = []
            self.isdetected = False
        else:
            coeffs = [0.0] * 6 
            self.isdetected = True

        if len(left_line) >= 3:
            lx = [p[0] for p in left_line]
            ly = [p[1] for p in left_line]
            
            model = np.polyfit(lx, ly, 2) 
            coeffs[0], coeffs[1], coeffs[2] = model[0], model[1], model[2]
        
        if len(right_line) >= 3:
            rx = [p[0] for p in right_line]
            ry = [p[1] for p in right_line]
            
            model = np.polyfit(rx, ry, 2)
            coeffs[3], coeffs[4], coeffs[5] = model[0], model[1], model[2]

        # 데이터 pub류
        msg = Float32MultiArray()
        msg.data = coeffs
        self.pub_coeffs.publish(msg)

        rospy.loginfo_throttle(1.0, 
        f"[Lane] Left:{len(left_line)} Right:{len(right_line)} -> Path Detected Check: {self.isdetected}")


    def run(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    RaidaLanePrediction().run()
