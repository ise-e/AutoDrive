import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

"""
라바콘 인식 분기는 한 라인의 3개의 라바콘을 찾아 2차 다항식을 그릴 수 있는지를 기준으로 정해짐.

라바콘 인식시 길이 6 리스트 publish.
인식 안될시 빈 배열 publish.
"""

class ObstacleDetector:
    def __init__(self):
        # 1. 구독자 및 발행자 설정
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.obj_pub = rospy.Publisher("/detected_obstacles", Float32MultiArray, queue_size=1)
        
        # 2. 파라미터 설정 (환경에 맞춰 튜닝)
        self.distance_threshold = 0.1  # 같은 물체로 판단할 점들 사이의 최대 거리 (m)
        self.min_cluster_size = 3      # 유효한 물체로 볼 최소 점 개수
        self.max_cluster_size = 30     # 라바콘 크기를 고려한 최대 점 개수
        self.roi_front_limit = 3.0     # 전방 탐색 제한 거리 (m) 

    def decide_center(self, points):
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
        
        # --- [1] 전처리: 극좌표를 직교좌표(x, y)로 변환 및 ROI 필터링 ---
        for i, r in enumerate(msg.ranges):
            # 무한대나 유효하지 않은 값 제외
            if r < 0.2 or r > self.roi_front_limit:
                continue
        
            angle = msg.angle_min + i * msg.angle_increment

            # [cite_start]전방 180도만 처리 (ROI 설정) [cite: 27]
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

        # --- [3] 클러스터별 좌표 및 크기(반경) 계산 ---
        obstacle_center_arr = []
        left_line = []
        right_line = []

        for cluster in clusters:
            arr = np.array(cluster)
            
            if len(arr) > 3:
                point = self.decide_center(arr)
                obstacle_center_arr.append(point)

        obstacle_center_arr = sorted(obstacle_center_arr, key=lambda x: x[2])

        for i, points in enumerate(obstacle_center_arr):
            if i > 1:
                dist_to_left = np.sum((right_line[-1]-point[:2])**2)
                dist_to_right = np.sum((left_line[-1]-point[:2])**2)

                if dist_to_left < dist_to_right:
                    left_line.append(point)
                else:
                    right_line.append(point)
            else:
                if point[1] > 0:
                    right_line.append(point)
                else:
                    left_line.append(point)

        # 포맷: [L_a, L_b, L_c,  R_a, R_b, R_c]
        if len(points) == 0:
            coeffs = []
        else:
            coeffs = [0.0] * 6 

        if len(left_line) >= 3:
            lx = [p[0] for p in left_line]
            ly = [p[1] for p in left_line]
            
            # [수정 2] 차수를 2로 변경 (y = ax^2 + bx + c)
            model = np.polyfit(lx, ly, 2) 
            coeffs[0], coeffs[1], coeffs[2] = model[0], model[1], model[2]
        
        if len(right_line) >= 3:
            rx = [p[0] for p in right_line]
            ry = [p[1] for p in right_line]
            
            model = np.polyfit(rx, ry, 2)
            coeffs[3], coeffs[4], coeffs[5] = model[0], model[1], model[2]

        # 한쪽만 보일 때 평행이동
        lane_width = 0.4
        
        if len(left_line) >= 3 and len(right_line) < 3:
            coeffs[3] = coeffs[0]            # a 복사
            coeffs[4] = coeffs[1]            # b 복사
            coeffs[5] = coeffs[2] - lane_width # c만 오른쪽으로 이동
            
        elif len(left_line) < 3 and len(right_line) >= 3:
            coeffs[0] = coeffs[3]            # a 복사
            coeffs[1] = coeffs[4]            # b 복사
            coeffs[2] = coeffs[5] + lane_width # c만 왼쪽으로 이동

        # --- [4] 데이터 퍼블리시 ---
        msg = Float64MultiArray()
        msg.data = coeffs
        self.pub_coeffs.publish(msg)

if __name__ == '__main__':
    rospy.init_node('obstacle_detector_node')
    node = ObstacleDetector()
    rospy.spin()