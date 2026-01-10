#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple, Optional

import numpy as np
import cv2
import rospy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

WIDTH, HEIGHT = 640, 480
CENTER = (WIDTH // 2, HEIGHT - 50)
SCALE = 80

class RaidaLanePrediction:
    """
    라바콘 인식시 길이 6 리스트 publish.
    인식 안될시 빈 배열 publish.
    """

    def __init__(self) -> None:
        rospy.init_node("raida_lane_prediction")

        # Params
        self.distance_threshold = 0.1  # 같은 물체로 판단할 점들 사이의 최대 거리 (m)
        self.min_cluster_size = 3      # 유효한 물체로 볼 최소 점 개수
        self.max_cluster_size = 30     # 라바콘 크기를 고려한 최대 점 개수
        self.roi_front_min_limit = 0.07       # 전방 최소 탐지 거리
        self.roi_front_max_limit = 1.3     # 전방 탐색 제한 거리 (m) 
        self.isdetected = False         # 라인 감지 확인용

        # 시각화용 데이터 저장 변수
        self.viz_points = []
        self.viz_cones_left = []
        self.viz_cones_right = []
        self.viz_coeffs = []

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
            if r < self.roi_front_min_limit or r > self.roi_front_max_limit:
                continue
        
            angle = msg.angle_min + i * msg.angle_increment

            # 전방 180도만 처리 (ROI 설정)
            if angle < -math.pi/2 or angle > math.pi/2:
                x = -r * math.cos(angle)
                y = -r * math.sin(angle)
                points.append([x, y])

        self.viz_points = points

        # 유클리드 클러스터링 알고리즘
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
                dist_to_left = np.sum((left_line[-1][0:2]-point[:2])**2)
                dist_to_right = np.sum((right_line[-1][0:2]-point[:2])**2)

                if dist_to_left < dist_to_right:
                    left_line.append(point)
                else:
                    right_line.append(point)
            else:
                if point[1] > 0:
                    left_line.append(point)
                else:
                    right_line.append(point)

        self.viz_cones_left = left_line
        self.viz_cones_right = right_line

        # 포맷: [L_a, L_b, L_c,  R_a, R_b, R_c]
        if len(points) == 0 or len(right_line) < 3 or len(left_line) < 3: 
            # 라바콘이 없거나 둘중 한쪽 차선의 라바콘이 2개 이하일 때, 장애물 없다고 판단.
            coeffs = []
            self.isdetected = False
        else:
            coeffs = [0.0] * 6 
            self.isdetected = True

        if self.isdetected:
            lx = [p[0] for p in left_line]
            ly = [p[1] for p in left_line]
            
            model = np.polyfit(lx, ly, 2) 
            coeffs[0], coeffs[1], coeffs[2] = model[0], model[1], model[2]
        
            rx = [p[0] for p in right_line]
            ry = [p[1] for p in right_line]
            
            model = np.polyfit(rx, ry, 2)
            coeffs[3], coeffs[4], coeffs[5] = model[0], model[1], model[2]

        self.viz_coeffs = coeffs

        # 데이터 pub류
        msg = Float32MultiArray()
        msg.data = coeffs
        self.pub_coeffs.publish(msg)

        rospy.loginfo_throttle(1.0, 
        f"[Lane] Left:{len(left_line)} Right:{len(right_line)} -> Path Detected Check: {self.isdetected}")


    def run(self) -> None:
        rate = rospy.Rate(30) # 30Hz
        while not rospy.is_shutdown():
            self.viz()
            rate.sleep()

    def viz(self):
        # 배경 생성 (검은색)
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        cv2.line(img, (CENTER[0], 0), (CENTER[0], HEIGHT), (60, 60, 60), 1)
        # 가로선 (차량 위치)
        cv2.line(img, (0, CENTER[1]), (WIDTH, CENTER[1]), (60, 60, 60), 1)

        # 2. Raw LiDAR Points 그리기
        # ROS 좌표: X(Forward), Y(Left)
        # 이미지 좌표: X(Right), Y(Down)
        # 변환: ImgX = CENTER[0] - Y*SCALE (좌우 반전 주의: ROS Y+는 왼쪽이므로 화면 왼쪽으로 가려면 빼야함)
        #       ImgY = CENTER[1] - X*SCALE (전방은 화면 위쪽이므로 빼야함)
        for (x, y) in self.viz_points:
            px = int(CENTER[0] - y * SCALE)  # Y(Left)를 가로축으로
            py = int(CENTER[1] - x * SCALE)  # X(Forward)를 세로축으로
            
            # 거리별 색상
            dist = math.sqrt(x*x + y*y)
            if dist < 0.3: color = (0, 0, 255)    # Red (Near)
            elif dist < 0.6: color = (0, 255, 255) # Yellow
            else: color = (0, 255, 0)              # Green (Far)

            # 화면 안에 들어오는 점만 그리기
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                cv2.circle(img, (px, py), 1, color, -1)

        # 3. 인식된 라바콘(Cluster Centers) 그리기
        # 왼쪽 콘: 파란색
        for p in self.viz_cones_left:
            px = int(CENTER[0] - p[1] * SCALE)
            py = int(CENTER[1] - p[0] * SCALE)
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                cv2.circle(img, (px, py), 5, (255, 0, 0), 2) # Blue Circle

        # 오른쪽 콘: 빨간색
        for p in self.viz_cones_right:
            px = int(CENTER[0] - p[1] * SCALE)
            py = int(CENTER[1] - p[0] * SCALE)
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                cv2.circle(img, (px, py), 5, (0, 0, 255), 2) # Red Circle

        # 4. 피팅된 차선 및 주행 경로(Line) 그리기
        if self.isdetected and len(self.viz_coeffs) == 6:
            la, lb, lc = self.viz_coeffs[0:3]
            ra, rb, rc = self.viz_coeffs[3:6]
            
            # 그릴 점들 계산 (전방 0m ~ 3.5m)
            x_vals = np.arange(0, 3.5, 0.1)
            
            pts_left = []
            pts_right = []
            pts_center = []

            for x in x_vals:
                # 다항식 계산: y = ax^2 + bx + c
                ly = la * x**2 + lb * x + lc  # 왼쪽 차선 Y좌표(Lateral)
                ry = ra * x**2 + rb * x + rc  # 오른쪽 차선 Y좌표(Lateral)
                cy = (ly + ry) / 2.0          # 중앙 경로

                # 이미지 좌표 변환
                # ImgX = CENTER[0] - Lateral * SCALE
                # ImgY = CENTER[1] - Forward * SCALE
                
                # 유효성 체크 (너무 멀리 튄 값 제외)
                if abs(ly) < 3.0:
                    pts_left.append([int(CENTER[0] - ly * SCALE), int(CENTER[1] - x * SCALE)])
                if abs(ry) < 3.0:
                    pts_right.append([int(CENTER[0] - ry * SCALE), int(CENTER[1] - x * SCALE)])
                if abs(cy) < 3.0:
                    pts_center.append([int(CENTER[0] - cy * SCALE), int(CENTER[1] - x * SCALE)])

            # 선 그리기 (Polylines)
            if len(pts_left) > 1 and (la!=0 or lb!=0):
                cv2.polylines(img, [np.array(pts_left, dtype=np.int32)], False, (255, 100, 0), 2) # 하늘색
            
            if len(pts_right) > 1 and (ra!=0 or rb!=0):
                cv2.polylines(img, [np.array(pts_right, dtype=np.int32)], False, (0, 100, 255), 2) # 주황색

            if len(pts_center) > 1:
                cv2.polylines(img, [np.array(pts_center, dtype=np.int32)], False, (0, 255, 0), 2) # 녹색 (주행경로)

        # 텍스트 정보 출력
        status = "Detected" if self.isdetected else "Searching..."
        cv2.putText(img, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 화면 출력
        cv2.imshow("LiDAR Lane Prediction", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    RaidaLanePrediction().run()