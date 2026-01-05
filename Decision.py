#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import numpy as np
from std_msgs.msg import String, Int16MultiArray, Float32MultiArray


class DecisionNode:
    def __init__(self):
        rospy.init_node('decision_node')

        # ---- 상태 ----
        self.STATE_DRIVE = "DRIVE"        # 차선 주행 + 장애물 회피
        self.STATE_WAIT = "WAIT"          # 노란선 정지 후 초록불 대기
        self.STATE_PARK = "PARKING"       # AR 태그 주차

        self.current_state = self.STATE_DRIVE

        # ---- 미션 / 시간 ----
        self.lap_count = 0
        self.last_stop_line_raw = "NONE"  # stop_line 엣지 감지용(직전 raw 값)
        self.stop_start_time = 0.0        # WAIT 진입 시각(대기 최소시간)

        # ---- 입력 데이터(콜백에서 최신값만 갱신) ----
        self.lane_coeffs = []             # 6개 실수: [좌(a,b,c), 우(a,b,c)]
        self.obstacles = []               # [x,y,...]
        self.stop_line_status = "NONE"    # NONE/WHITE/YELLOW
        self.traffic_light = "RED"        # RED/GREEN
        self.ar_tag = None                # [id, dist, angle(rad)]

        # ---- 구독자 ----
        rospy.Subscriber("/lane_coeffs", Float32MultiArray, self.lane_callback)
        rospy.Subscriber("/detected_obstacles", Float32MultiArray, self.obstacle_callback)
        rospy.Subscriber("/stop_line_status", String, self.stop_line_callback)
        rospy.Subscriber("/traffic_light_status", String, self.light_callback)
        rospy.Subscriber("/ar_tag_info", Float32MultiArray, self.ar_callback)

        # ---- 발행자 ----
        self.motor_pub = rospy.Publisher("/motor", Int16MultiArray, queue_size=1)          # [steer,speed]
        self.dir_pub = rospy.Publisher("/mission_direction", String, queue_size=1)         # LEFT/RIGHT
        self.status_pub = rospy.Publisher("/mission_status", String, queue_size=1)         # STOP/PARKING

    # ===== 콜백: 최신 데이터만 저장 =====
    def lane_callback(self, msg):     self.lane_coeffs = msg.data
    def obstacle_callback(self, msg): self.obstacles = msg.data
    def light_callback(self, msg):    self.traffic_light = msg.data
    def ar_callback(self, msg):       self.ar_tag = msg.data

    def stop_line_callback(self, msg):
        # stop_line은 "이벤트"라서, 값 변화(엣지)에서만 1회 처리한다.
        raw = msg.data
        if raw == self.last_stop_line_raw:
            return
        self.last_stop_line_raw = raw

        # NONE -> (WHITE/YELLOW)로 바뀌는 순간만 이벤트로 취급
        if raw == "NONE":
            return

        self.stop_line_status = raw

        if raw == "WHITE":
            self.lap_count += 1

    # ===== 차선 조향(P 제어) =====
    def get_steering(self):
        if len(self.lane_coeffs) < 6:
            return 90

        la, lb, lc, ra, rb, rc = self.lane_coeffs
        y = 400
        left_x = la * y**2 + lb * y + lc
        right_x = ra * y**2 + rb * y + rc
        center_x = (left_x + right_x) * 0.5

        err = center_x - 320.0           # 화면 폭 640 기준 중앙(320)
        return int(90 + err * 0.15)      # 조향 중립=90

    # ===== 메인 판단(step): 30Hz =====
    def step(self):
        # 초반 갈림길 포함, 방향 지시는 항상 최신값을 발행한다.
        # (수신 노드가 latched가 아니거나, 늦게 켜져도 안정적으로 동작)
        direction = "LEFT" if self.lap_count >= 2 else "RIGHT"
        self.dir_pub.publish(direction)
        
        steer = self.get_steering()
        speed = 100

        if self.current_state == self.STATE_DRIVE:
            # 장애물 회피(DRIVE에서만)
            if len(self.obstacles) >= 3:
                x, y = self.obstacles[0], self.obstacles[1]
                if np.hypot(x, y) < 1.0:
                    steer += 25

            # 노란선 정지 조건: 출발 직후(0바퀴) 또는 2바퀴 이후
            is_mission_stop = (self.lap_count == 0 or self.lap_count >= 2)
            if self.stop_line_status == "YELLOW" and is_mission_stop:
                self.current_state = self.STATE_WAIT
                self.stop_start_time = time.time()
                self.stop_line_status = "NONE"     # 이벤트 소모(재진입 방지)
                self.status_pub.publish("STOP")

        elif self.current_state == self.STATE_WAIT:
            speed = 0
            # 정지 후 1초 이상 대기, 초록불이면 출발/전이
            if self.traffic_light == "GREEN" and (time.time() - self.stop_start_time) > 1.0:
                self.current_state = self.STATE_PARK if self.lap_count >= 2 else self.STATE_DRIVE

        elif self.current_state == self.STATE_PARK:
            self.status_pub.publish("PARKING")
            if self.ar_tag:
                dist, ang = self.ar_tag[1], self.ar_tag[2]
                if dist < 0.3:
                    speed, steer = 0, 90
                else:
                    speed = 95
                    steer = 90 + int(np.degrees(ang))

        return steer, speed

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            s, v = self.step()
            self.motor_pub.publish(Int16MultiArray(data=[int(s), int(v)]))
            rate.sleep()


if __name__ == "__main__":
    DecisionNode().run()
