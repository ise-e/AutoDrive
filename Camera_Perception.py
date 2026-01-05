#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge

class IntegratedPerception:
    def __init__(self):
        self.bridge = CvBridge()
        
        # --- [1] 발행자(Publisher) 설정 ---
        self.ar_pub = rospy.Publisher("/ar_tag_info", Float32MultiArray, queue_size=1)
        self.traffic_pub = rospy.Publisher("/traffic_light_status", String, queue_size=1)
        self.lane_pub = rospy.Publisher("/lane_coeffs", Float32MultiArray, queue_size=1)
        self.stop_line_pub = rospy.Publisher("/stop_line_status", String, queue_size=1)
        
        # --- [2] 구독자(Subscriber) 설정 ---
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.mission_direction_sub = rospy.Subscriber("/mission_direction", String, self.mission_direction_callback)
        self.mission_status_sub = rospy.Subscriber("/mission_status", String, self.mission_status_callback)
        
        # --- [3] 파라미터 및 초기화 설정 ---
        self.current_mission_direction = "RIGHT"
        self.current_mission_status = "NONE"
        
        # AR 태그 설정
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.tag_size = 0.15
        self.focal_length = 600.0
        
        # 신호등 및 차선 HSV 범위 (공통 관리)
        self.lower_red1 = np.array([0, 100, 100]);   self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100]); self.upper_red2 = np.array([179, 255, 255])
        self.lower_yellow = np.array([15, 100, 100]); self.upper_yellow = np.array([35, 255, 255])
        self.lower_green = np.array([45, 100, 100]);  self.upper_green = np.array([90, 255, 255])
        self.lower_white = np.array([0, 0, 200]);    self.upper_white = np.array([179, 40, 255])

        # BEV 변환 행렬
        self.src_pts = np.float32([[200, 300], [440, 300], [50, 450], [590, 450]])
        self.dst_pts = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def mission_direction_callback(self, msg):
        self.current_mission_direction = msg.data.upper()

    def mission_status_callback(self, msg):
        self.current_mission_status = msg.data.upper()

    # AR 태그 검출 및 정보 발행
    def process_ar_tag(self, gray_frame):
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            ar_data = []
            for i in range(len(ids)):
                c = corners[i][0]
                pixel_width = np.linalg.norm(c[0] - c[1])
                distance = (self.tag_size * self.focal_length) / pixel_width
                
                img_center_x = gray_frame.shape[1] / 2
                pixel_error = np.mean(c[:, 0]) - img_center_x
                angle = np.arctan2(pixel_error, self.focal_length)
                
                ar_data.extend([float(ids[i]), float(distance), float(angle)])
            
            self.ar_pub.publish(Float32MultiArray(data=ar_data))

    # 신호등 색상 검출 및 정보 발행
    def process_traffic_light(self, hsv_frame):
        h, w = hsv_frame.shape[:2]
        roi_hsv = hsv_frame[0:int(h*0.4), int(w*0.2):int(w*0.8)]
        
        mask_red = cv2.bitwise_or(cv2.inRange(roi_hsv, self.lower_red1, self.upper_red1),
                                  cv2.inRange(roi_hsv, self.lower_red2, self.upper_red2))
        mask_yellow = cv2.inRange(roi_hsv, self.lower_yellow, self.upper_yellow)
        mask_green = cv2.inRange(roi_hsv, self.lower_green, self.upper_green)
        
        counts = {
            'RED': cv2.countNonZero(mask_red),
            'YELLOW': cv2.countNonZero(mask_yellow),
            'GREEN': cv2.countNonZero(mask_green)
        }
        
        best_color = max(counts, key=counts.get)
        status = best_color if counts[best_color] > 200 else "UNKNOWN"
        self.traffic_pub.publish(status)

    # BEV 이미지에서 차선 계수 및 정지선 상태 검출
    def process_lanes_and_stopline(self, bev_frame):
        hsv_bev = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2HSV)
        
        # 정지선 검출 로직
        y_mask = cv2.inRange(hsv_bev, self.lower_yellow, self.upper_yellow)
        w_mask = cv2.inRange(hsv_bev, self.lower_white, self.upper_white)
        
        h, w = y_mask.shape
        check_area = slice(int(h*0.7), int(h*0.8))
        y_pixels = np.sum(y_mask[check_area, :]) / 255
        w_pixels = np.sum(w_mask[check_area, :]) / 255
        
        stop_status = "NONE"
        if y_pixels > w * 15: stop_status = "YELLOW"
        elif w_pixels > w * 15: stop_status = "WHITE"
        self.stop_line_pub.publish(stop_status)

        # 차선 피팅 (슬라이딩 윈도우)
        combined_mask = cv2.bitwise_or(w_mask, y_mask)
        edges = cv2.Canny(combined_mask, 50, 150)
        
        left_fit, right_fit = self.perform_sliding_window(edges)
        if left_fit is not None and right_fit is not None:
            self.lane_pub.publish(Float32MultiArray(data=list(left_fit) + list(right_fit)))

    def perform_sliding_window(self, binary_img):
        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
        midpoint = len(histogram) // 2
        
        if self.current_mission_direction == "RIGHT":
            leftx_base = np.argmax(histogram[midpoint-100:midpoint+100]) + (midpoint-100)
            rightx_base = np.argmax(histogram[midpoint+150:]) + (midpoint+150)
        elif self.current_mission_direction == "LEFT":
            leftx_base = np.argmax(histogram[:midpoint-150])
            rightx_base = np.argmax(histogram[midpoint-100:midpoint+100]) + (midpoint-100)
        else:
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = binary_img.shape[0] // nwindows
        margin = 60 # 탐색 너비 확대
        minpix = 40 # 유효 픽셀 최소 기준

        nonzero = binary_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        left_lane_inds, right_lane_inds = [], []
        lx_curr, rx_curr = leftx_base, rightx_base

        for window in range(nwindows):
            y_low, y_high = binary_img.shape[0] - (window+1)*window_height, binary_img.shape[0] - window*window_height
            
            # 윈도우 바운더리 계산 및 인덱스 추출
            win_lx_l, win_lx_h = lx_curr - margin, lx_curr + margin
            win_rx_l, win_rx_h = rx_curr - margin, rx_curr + margin

            good_l = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= win_lx_l) & (nonzerox < win_lx_h)).nonzero()[0]
            good_r = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= win_rx_l) & (nonzerox < win_rx_h)).nonzero()[0]

            left_lane_inds.append(good_l)
            right_lane_inds.append(good_r)

            if len(good_l) > minpix: lx_curr = int(np.mean(nonzerox[good_l]))
            if len(good_r) > minpix: rx_curr = int(np.mean(nonzerox[good_r]))

        try:
            l_inds = np.concatenate(left_lane_inds)
            r_inds = np.concatenate(right_lane_inds)
            
            # 데이터가 부족할 경우에 대한 방어 로직
            if len(l_inds) < 500 or len(r_inds) < 500:
                return None, None
                
            l_fit = np.polyfit(nonzeroy[l_inds], nonzerox[l_inds], 2)
            r_fit = np.polyfit(nonzeroy[r_inds], nonzerox[r_inds], 2)
            return l_fit, r_fit
        except:
            return None, None

    def image_callback(self, msg):
        # 1. 공통 전처리: 1회만 수행하여 자원 절약
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (640, 480))
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 2. 각 알고리즘에 필요한 변환본 생성
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        bev = cv2.warpPerspective(blur, self.M, (640, 480))
        
        self.process_lanes_and_stopline(bev)
        
        # 3. 개별 프로세스 실행
        if self.current_mission_status == "STOP":
            self.process_traffic_light(hsv)
        if self.current_mission_status == "PARKING":
            self.process_ar_tag(gray)

if __name__ == '__main__':
    rospy.init_node('integrated_perception_node')
    IntegratedPerception()
    rospy.spin()