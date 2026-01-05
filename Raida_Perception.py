import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

class ObstacleDetector:
    def __init__(self):
        # 1. 구독자 및 발행자 설정
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.obj_pub = rospy.Publisher("/detected_obstacles", Float32MultiArray, queue_size=1)
        
        # 2. 파라미터 설정 (환경에 맞춰 튜닝)
        self.distance_threshold = 0.1  # 같은 물체로 판단할 점들 사이의 최대 거리 (m)
        self.min_cluster_size = 3      # 유효한 물체로 볼 최소 점 개수
        self.max_cluster_size = 30     # 라바콘 크기를 고려한 최대 점 개수
        [cite_start]self.roi_front_limit = 3.0     # 전방 탐색 제한 거리 (m) 

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
            if -math.pi/2 < angle < math.pi/2:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append([x, y])

        if len(points) == 0:
            return

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
        obstacle_data = []
        for cluster in clusters:
            arr = np.array(cluster)
            center_x = np.mean(arr[:, 0])
            center_y = np.mean(arr[:, 1])
            
            # 중심점에서 가장 먼 점까지의 거리를 반경으로 정의
            distances = np.sqrt((arr[:, 0] - center_x)**2 + (arr[:, 1] - center_y)**2)
            radius = np.max(distances)
            
            # 데이터 추가: [x, y, radius]
            obstacle_data.extend([center_x, center_y, radius])

        # --- [4] 데이터 퍼블리시 ---
        if obstacle_data:
            msg_to_pub = Float32MultiArray()
            msg_to_pub.data = obstacle_data
            self.obj_pub.publish(msg_to_pub)

if __name__ == '__main__':
    rospy.init_node('obstacle_detector_node')
    node = ObstacleDetector()
    rospy.spin()