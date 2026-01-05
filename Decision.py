import rospy
import numpy as np
from std_msgs.msg import String, Int16MultiArray, Float32MultiArray

class DecisionNode:
    def __init__(self):
        # rospy.init_node('decision_node') # runtime_node에서 초기화하므로 삭제함

        # --- 상태 정의 (UC8 기준) ---
        self.STATE_START_WAIT = 0   # 출발 대기
        self.STATE_YELLOW_STOP_1 = 1 # 1차 노란선 정지 (UC1)
        self.STATE_LANE_FOLLOW = 2   # 기본 주행 (UC2, UC3)
        self.STATE_WHITE_COUNT_1 = 3 # 1차 흰색선 통과 (UC4)
        self.STATE_WHITE_COUNT_2 = 4 # 2차 흰색선 감지 및 왼쪽 진입 (UC5)
        self.STATE_YELLOW_STOP_2 = 5 # 2차 노란선 정지 (UC6)
        self.STATE_PARKING = 6       # AR 태그 주차 (UC7)
        self.STATE_FINISH = 7        # 종료

        self.current_state = self.STATE_START_WAIT
        
        # --- 데이터 저장 변수 ---
        self.lane_coeffs = None
        self.stop_line_status = "NONE"
        self.traffic_light = "UNKNOWN"
        self.obstacles = []
        self.ar_tag = None # [ID, Distance, Angle]
        
        self.stop_line_count = 0
        self.last_stop_line_time = rospy.Time.now()

        # --- 발행자/구독자 설정 ---
        self.motor_pub = rospy.Publisher("/motor", Int16MultiArray, queue_size=1)
        self.mission_direction_pub = rospy.Publisher("/mission_direction", String, queue_size=1)
        self.mission_status_pub = rospy.Publisher("/mission_status", String, queue_size=1)

        rospy.Subscriber("/lane_coeffs", Float32MultiArray, self.lane_cb)
        rospy.Subscriber("/stop_line_status", String, self.stop_line_cb)
        rospy.Subscriber("/traffic_light_status", String, self.light_cb)
        rospy.Subscriber("/detected_obstacles", Float32MultiArray, self.obs_cb)
        rospy.Subscriber("/ar_tag_info", Float32MultiArray, self.ar_cb)

    # --- 콜백 함수들 ---
    def lane_cb(self, msg): self.lane_coeffs = msg.data
    def light_cb(self, msg): self.traffic_light = msg.data
    def obs_cb(self, msg): self.obstacles = msg.data
    def ar_cb(self, msg): self.ar_tag = msg.data

    def stop_line_cb(self, msg):
        # 중복 카운트 방지 (2초 쿨타임)
        if msg.data != "NONE" and (rospy.Time.now() - self.last_stop_line_time).to_sec() > 2.0:
            self.stop_line_status = msg.data
            self.stop_line_count += 1
            self.last_stop_line_time = rospy.Time.now()
            rospy.loginfo(f"Line Detected: {msg.data}, Total Count: {self.stop_line_count}")

    # --- 제어 로직 ---
    # 차선 계수를 이용한 목표 조향각 계산 (단순화된 PID)
    def get_steer_from_lane(self):
        if self.lane_coeffs is None or len(self.lane_coeffs) < 6:
            return 90
        # 차선 중앙점 계산 (y=400 지점의 x값 사용)
        l_a, l_b, l_c = self.lane_coeffs[0:3]
        r_a, r_b, r_c = self.lane_coeffs[3:6]
        y_eval = 400
        center_x = ((l_a*y_eval**2 + l_b*y_eval + l_c) + (r_a*y_eval**2 + r_b*y_eval + r_c)) / 2
        error = center_x - 320 # 640 해상도 기준 중앙 오차
        return int(90 + (error * 0.15)) # Kp 게인 조절 필요

    def step(self):
        steer = 90
        speed = 90 

        # --- [UC8] 트랙 주행 시나리오 제어 ---
        
        # [UC1] 1차 노란색 실선 정지 및 출발
        if self.current_state == self.STATE_START_WAIT:
            # UC1-5: 출발 시 기본 방향을 오른쪽 레인으로 설정
            self.mission_direction_pub.publish("RIGHT") 
            speed = 100 
            self.current_state = self.STATE_YELLOW_STOP_1

        elif self.current_state == self.STATE_YELLOW_STOP_1:
            if self.stop_line_status == "YELLOW":
                speed = 90
                self.mission_status_pub.publish("STOP")
                # UC1-4: 신호등 초록불 인지 시 출발
                if self.traffic_light == "GREEN":
                    self.current_state = self.STATE_LANE_FOLLOW
            else:
                speed = 98; steer = self.get_steer_from_lane()

        # [UC2, UC3] 레인 주행 및 콘 장애물 회피
        elif self.current_state == self.STATE_LANE_FOLLOW:
            speed = 100; steer = self.get_steer_from_lane()
            
            # [UC3] 장애물 회피 (라이다 데이터 활용)
            if len(self.obstacles) >= 3:
                x, y = self.obstacles[0], self.obstacles[1]
                dist = np.sqrt(x**2 + y**2)
                if dist < 1.0:
                    steer += 25

            # [UC4] 1차 흰색 실선 감지 (한바퀴 순환)
            if self.stop_line_status == "WHITE" and self.stop_line_count == 1:
                rospy.loginfo("UC4: First Lap Completed. Repeat Lane Follow.")
                self.mission_direction_pub.publish("RIGHT") # 계속 오른쪽 레인 유지

            # [UC5] 2차 흰색 실선 감지 (왼쪽 진입)
            if self.stop_line_status == "WHITE" and self.stop_line_count == 2:
                rospy.loginfo("UC5: Second White Line. Transition to LEFT Lane.")
                self.mission_direction_pub.publish("LEFT") # 왼쪽 레인 진입 명령
                self.current_state = self.STATE_WHITE_COUNT_2

        # [UC6] 2차 노란색 실선 감지
        elif self.current_state == self.STATE_WHITE_COUNT_2:
            speed = 98; steer = self.get_steer_from_lane()
            if self.stop_line_status == "YELLOW":
                speed = 90
                self.mission_status_pub.publish("STOP")
                # UC6-4: 초록불 대기
                if self.traffic_light == "GREEN":
                    self.current_state = self.STATE_PARKING
                    self.mission_status_pub.publish("PARKING")

        # [UC7] AR 태그 인식 주차
        elif self.current_state == self.STATE_PARKING:
            if self.ar_tag is not None:
                id, dist, angle = self.ar_tag[0:3]
                if dist > 0.3:
                    speed = 95; steer = 90 + int(np.degrees(angle))
                else:
                    speed = 90
                    rospy.loginfo("UC7: Parking Completed.")
                    self.current_state = self.STATE_FINISH

        # 최종 명령 발행
        return int(steer), int(speed)