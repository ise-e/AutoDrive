#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decision.py
[2026-01-04] Refactored for Stability & Safety

주요 개선점:
1. Thread-Safety: SensorHub를 통해 데이터 스냅샷(Snapshot) 사용 -> 센서 데이터 꼬임 방지
2. Coordinate Normalization: 카메라(Pixel)와 라이다(Meter) 에러를 -1.0~1.0으로 통일
3. Robust FSM: 상태 전이 로직을 분리하여 관리 (가독성 향상)
4. Safety Watchdog: 센서 끊김 시 관성 주행(0.5s) 후 비상 정지(1.5s)
"""

from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum, auto

import rospy
from std_msgs.msg import Int16MultiArray, String, Float32MultiArray

# ==============================================================================
# 1. Configuration (튜닝 포인트 통합 관리)
# ==============================================================================
@dataclass
class Config:
    # ---------------- 하드웨어 설정 ----------------
    steer_center: int = 90      # 조향 중앙값 (직진 PWM)
    steer_min: int = 50         # 조향 최소값 (왼쪽 최대)
    steer_max: int = 130        # 조향 최대값 (오른쪽 최대)
    
    # ---------------- 속도 설정 ----------------
    speed_drive: int = 100      # 평시 주행 속도
    speed_caution: int = 95     # 코너링/라바콘 주행 속도
    speed_parking: int = 98     # 주차 진입 속도
    speed_min: int = 0          # 정지

    # ---------------- PID 제어 (조향 감도) ----------------
    kp: float = 120.0
    ki: float = 0.05
    kd: float = 250.0
    

    # 적분(Integral) 안전장치
    integ_limit: float = 0.25         # |I| 최대 (정규화 단위)
    integ_reset_err: float = 0.02     # |err|가 이하면 I 리셋

    # 조향 부호(차량/서보 방향에 따라 뒤집힘). True면 조향 출력 부호를 반전
    steer_invert: bool = False

    # AR 태그 추종 조향 게인 (deg -> PWM 변환 스케일)
    ar_steer_gain: float = 1.0

    # 센서 퓨전 신뢰도/신선도 파라미터
    cam_ttl_sec: float = 0.25
    lidar_ttl_sec: float = 0.25
    cam_single_lane_weight: float = 0.65   # 3 coeffs일 때 신뢰도
    cam_double_lane_weight: float = 1.00   # 6 coeffs일 때 신뢰도
    lidar_double_lane_weight: float = 1.00 # 6 coeffs일 때 신뢰도
    lidar_single_lane_weight: float = 0.25 # 한쪽만 보이면 보수적으로 반영

    # 갈림길/단일차선에서 미션 방향으로 "약한 바이어스" (PWM 단위, 기본 0=OFF)
    fork_bias_pwm: int = 0
    # ---------------- 센서 파라미터 ----------------
    # 카메라 (Pixel 단위)
    cam_width: int = 640
    cam_y_far: int = 150        # 원거리 점 (이미지 상단)
    cam_y_near: int = 400       # 근거리 점 (이미지 하단)
    w_cam_far: float = 0.6      # 원거리 가중치
    w_cam_near: float = 0.4     # 근거리 가중치
    
    # 라이다 (Meter 단위)
    lidar_road_width_m: float = 0.2  # 정규화를 위한 도로 반폭 기준 (m) : 차선폭 0.4m -> 반폭 0.2m  # 정규화를 위한 도로 반폭 기준 (m)
    obs_eval_x: float = 0.5          # 라바콘 회피 판단 전방 거리 (m)

    # ---------------- 안전 장치 ----------------
    watchdog_soft_sec: float = 0.5   # 관성 주행 허용 시간
    watchdog_hard_sec: float = 1.5   # 비상 정지 발동 시간
    park_stop_dist: float = 0.3      # 주차 정지 거리 (m)

# ==============================================================================
# 2. Data Structures (Immutable Snapshots)
# ==============================================================================
@dataclass(frozen=True)
class CamLane:
    coeffs: Tuple[float, ...]
    timestamp: float

    def get_error_norm(self, cfg: Config) -> float:
        """
        카메라 좌표계(Pixel) 에러를 -1.0(Left) ~ 1.0(Right) 범위로 정규화
        수식: x = ay^2 + by + c
        """
        if not self.coeffs: return 0.0
        
        # Dual Lane (6 coeffs) or Single Lane (3 coeffs)
        if len(self.coeffs) >= 6:
            la, lb, lc, ra, rb, rc = self.coeffs[:6]
            # 왼쪽/오른쪽 차선의 x좌표 계산
            lx_far = la*cfg.cam_y_far**2 + lb*cfg.cam_y_far + lc
            rx_far = ra*cfg.cam_y_far**2 + rb*cfg.cam_y_far + rc
            cx_far = (lx_far + rx_far) / 2.0
            
            lx_near = la*cfg.cam_y_near**2 + lb*cfg.cam_y_near + lc
            rx_near = ra*cfg.cam_y_near**2 + rb*cfg.cam_y_near + rc
            cx_near = (lx_near + rx_near) / 2.0
        else:
            # Single Lane
            a, b, c = self.coeffs[:3]
            cx_far = a*cfg.cam_y_far**2 + b*cfg.cam_y_far + c
            cx_near = a*cfg.cam_y_near**2 + b*cfg.cam_y_near + c

        img_center = cfg.cam_width / 2.0
        err_far = (cx_far - img_center)
        err_near = (cx_near - img_center)
        
        # 픽셀 에러 가중 합산
        err_pixel = (err_far * cfg.w_cam_far) + (err_near * cfg.w_cam_near)
        
        # 정규화 (도로 폭의 절반을 약 200px로 가정)
        return max(-1.0, min(1.0, err_pixel / 200.0))

@dataclass(frozen=True)
class LidarLane:
    coeffs: Tuple[float, ...]
    timestamp: float

    def get_error_norm(self, cfg: Config) -> float:
        """
        라이다 좌표계(Meter) 에러를 -1.0 ~ 1.0 범위로 정규화
        수식: y = ax^2 + bx + c (차량 전방이 x축)
        """
        if not self.coeffs or len(self.coeffs) < 6: return 0.0
        
        # [La, Lb, Lc, Ra, Rb, Rc]
        la, lb, lc = self.coeffs[:3]
        ra, rb, rc = self.coeffs[3:6]
        
        eval_x = cfg.obs_eval_x
        
        # y coordinates (Lateral deviation)
        ly = la*eval_x**2 + lb*eval_x + lc
        ry = ra*eval_x**2 + rb*eval_x + rc
        cy = (ly + ry) / 2.0
        
        # 정규화
        return max(-1.0, min(1.0, cy / cfg.lidar_road_width_m))

@dataclass
class WorldState:
    timestamp: float
    cam: Optional[CamLane] = None
    lidar: Optional[LidarLane] = None
    stop_sign: str = "NONE"
    traffic_light: str = "UNKNOWN"
    ar_dist: Optional[float] = None
    ar_angle: Optional[float] = None

# ==============================================================================
# 3. Sensor Hub (Thread-Safe Manager)
# ==============================================================================
class SensorHub:
    def __init__(self):
        self._lock = threading.RLock()
        self._cam: Optional[CamLane] = None
        self._lidar: Optional[LidarLane] = None
        self._stop = "NONE"
        self._light = "UNKNOWN"
        self._ar_dist = None
        self._ar_angle = None
        
    def update_cam(self, data):
        with self._lock:
            self._cam = CamLane(tuple(data), time.time())
    
    def update_lidar(self, data):
        with self._lock:
            self._lidar = LidarLane(tuple(data), time.time())

    def update_stop(self, status):
        with self._lock:
            self._stop = (status or "NONE").upper()

    def update_light(self, status):
        with self._lock:
            self._light = (status or "UNKNOWN").upper()

    def update_ar(self, dist, angle):
        with self._lock:
            self._ar_dist = dist
            self._ar_angle = angle

    def get_snapshot(self) -> WorldState:
        with self._lock:
            return WorldState(
                timestamp=time.time(),
                cam=self._cam,
                lidar=self._lidar,
                stop_sign=self._stop,
                traffic_light=self._light,
                ar_dist=self._ar_dist,
                ar_angle=self._ar_angle
            )

# ==============================================================================
# 4. Vehicle Controller (PID & Actuation)
# ==============================================================================
class VehicleController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_err = 0.0
        self.integ_err = 0.0
        self.prev_output = 0.0  # Smoothing용

    def compute_steer(self, error_norm: float, dt: float) -> int:
        """
        error_norm: -1.0 ~ 1.0
        return: Steer PWM (절대값, 중앙값 포함)
        """
        if dt <= 0: dt = 0.033

        # 1. PID Terms
        p_term = error_norm * self.cfg.kp
        
        # Anti-Windup (I-term)
        # 작은 오차 구간에서는 I를 빠르게 비워서 바이어스/드리프트를 줄입니다.
        if abs(error_norm) <= self.cfg.integ_reset_err:
            self.integ_err = 0.0
        else:
            self.integ_err += error_norm * dt
            lim = float(self.cfg.integ_limit)
            self.integ_err = max(-lim, min(lim, self.integ_err))
        i_term = self.integ_err * self.cfg.ki
        
        d_term = ((error_norm - self.prev_err) / dt) * self.cfg.kd
        
        raw_output = p_term + i_term + d_term
        
        # 2. Output Smoothing (EMA Filter)
        alpha = 0.6
        smoothed_output = alpha * raw_output + (1 - alpha) * self.prev_output
        
        self.prev_err = error_norm
        self.prev_output = smoothed_output
        
        # 3. Final Calculation
        # 기본 규약: error_norm > 0 (우측 치우침) -> 좌회전 방향으로 보정
        # 차량/서보 방향이 반대면 ~steer_invert:=1 로 뒤집어 주세요.
        if self.cfg.steer_invert:
            steer = self.cfg.steer_center + int(smoothed_output)
        else:
            steer = self.cfg.steer_center - int(smoothed_output)
        return self._clamp(steer)

    def _clamp(self, val: int) -> int:
        return max(self.cfg.steer_min, min(self.cfg.steer_max, int(val)))
        
    def reset_pid(self):
        self.prev_err = 0.0
        self.integ_err = 0.0
        self.prev_output = 0.0

# ==============================================================================
# 5. Mission FSM (Logic)
# ==============================================================================
class MissionState(Enum):
    START = auto()
    CHECK_STOP_1 = auto()    # 첫 상차구역 (Yellow)
    DRIVE_1 = auto()         # 1바퀴째 주행 (우측통행)
    DRIVE_2 = auto()         # 2바퀴째 주행 (좌측통행)
    CHECK_STOP_2 = auto()    # 하차구역 (Yellow)
    PARKING_SEARCH = auto()  # 주차구역 진입
    PARKING_EXEC = auto()    # AR 태그 발견 및 주차
    FINISH = auto()

class MissionManager:
    def __init__(self):
        self.state = MissionState.START
        self.direction = "RIGHT" # 초기 우측통행
        self.white_cnt = 0
        self.last_state_change = time.time()
        self.debounce_time = 2.0 # 상태 변경 후 2초간 중복 인식 방지

    def update(self, s: WorldState) -> str:
        now = s.timestamp
        # Debounce 체크
        if now - self.last_state_change < self.debounce_time:
            return self._status_str()

        # --- 상태 전이 로직 ---
        if self.state == MissionState.START:
            self._to(MissionState.CHECK_STOP_1, now)
            
        elif self.state == MissionState.CHECK_STOP_1:
            # 노란선 정지 상태에서 초록불이면 출발
            if s.stop_sign == "YELLOW" and s.traffic_light == "GREEN":
                self._to(MissionState.DRIVE_1, now)
        
        elif self.state == MissionState.DRIVE_1:
            # 흰색 정지선 카운트
            if s.stop_sign == "WHITE":
                self.white_cnt += 1
                self.last_state_change = now # 카운트 중복 방지
                rospy.loginfo(f"[FSM] White Line Count: {self.white_cnt}")
                
                # 2바퀴 돌았으면 좌측 통행(하차구역 진입) 준비
                if self.white_cnt >= 2:
                    self.direction = "LEFT"
                    self._to(MissionState.DRIVE_2, now)
        
        elif self.state == MissionState.DRIVE_2:
             # 좌측 주행 중 노란선 만나면 하차 구역
             if s.stop_sign == "YELLOW":
                 self._to(MissionState.CHECK_STOP_2, now)

        elif self.state == MissionState.CHECK_STOP_2:
            # 하차 완료(초록불) -> 주차 구역 이동
            if s.stop_sign == "YELLOW" and s.traffic_light == "GREEN":
                self._to(MissionState.PARKING_SEARCH, now)

        elif self.state == MissionState.PARKING_SEARCH:
            # AR 태그 발견 시 주차 실행 모드
            if s.ar_dist is not None:
                self._to(MissionState.PARKING_EXEC, now)

        elif self.state == MissionState.PARKING_EXEC:
            # 주차 거리 도달
            if s.ar_dist is not None and s.ar_dist <= 0.3: # Config 값 참조 필요
                self._to(MissionState.FINISH, now)

        return self._status_str()

    def _to(self, new_state, now):
        rospy.loginfo(f"[FSM] State Change: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.last_state_change = now

    def _status_str(self):
        if "CHECK_STOP" in self.state.name: return "STOP"
        if "PARKING" in self.state.name: return "PARKING"
        if self.state == MissionState.FINISH: return "FINISH"
        return "NONE"

# ==============================================================================
# 6. Main Node
# ==============================================================================
class DecisionNode:

    def _fresh_weight(self, age_sec: float, ttl_sec: float) -> float:
        if age_sec < 0:
            return 0.0
        if ttl_sec <= 1e-6:
            return 0.0
        # 0~ttl 구간에서 1 -> 0 으로 선형 감소
        w = 1.0 - (age_sec / ttl_sec)
        return max(0.0, min(1.0, w))

    def _lane_reliability(self, coeffs, single_w: float, double_w: float) -> float:
        if coeffs is None:
            return 0.0
        n = len(coeffs)
        if n >= 6:
            return float(double_w)
        if n >= 3:
            return float(single_w)
        return 0.0
    def __init__(self):
        rospy.init_node("decision_node")
        
        # 1. Config 로드 (ROS Param으로 덮어쓰기 가능)
        self.cfg = Config(
            steer_center=int(rospy.get_param("~steer_center", 90)),
            lidar_road_width_m=float(rospy.get_param("~lidar_road_width_m", 0.2)),
            kp=float(rospy.get_param("~kp", 120.0)),
            kd=float(rospy.get_param("~kd", 250.0)),
            speed_drive=int(rospy.get_param("~speed_drive", 100)),
            park_stop_dist=float(rospy.get_param("~park_stop_dist", 0.3)),
        )
        
        # 2. Modules
        self.hub = SensorHub()
        self.ctrl = VehicleController(self.cfg)
        self.mission = MissionManager()
        
        # 3. Pubs
        self.motor_topic = rospy.get_param("~motor_topic", "/cmd/motor")
        self.pub_motor = rospy.Publisher(self.motor_topic, Int16MultiArray, queue_size=1)
        self.pub_dir = rospy.Publisher("/mission_direction", String, queue_size=1)
        self.pub_status = rospy.Publisher("/mission_status", String, queue_size=1)
        
        # 4. Subs
        rospy.Subscriber("/lane_coeffs", Float32MultiArray, lambda m: self.hub.update_cam(m.data))
        rospy.Subscriber("/obs_lane_coeffs", Float32MultiArray, lambda m: self.hub.update_lidar(m.data))
        rospy.Subscriber("/stop_line_status", String, lambda m: self.hub.update_stop(m.data))
        rospy.Subscriber("/traffic_light_status", String, lambda m: self.hub.update_light(m.data))
        rospy.Subscriber("/ar_tag_info", Float32MultiArray, self._cb_ar)

        self.last_loop_time = time.time()
        rospy.loginfo("[Decision] System Ready.")
        rospy.loginfo(f"[Decision] lidar_road_width_m(half width)={self.cfg.lidar_road_width_m:.3f}m")

    def _cb_ar(self, msg):
        if msg.data and len(msg.data) >= 3:
            # msg.data = [id, dist, ang]
            self.hub.update_ar(msg.data[1], msg.data[2])

    def run(self):
        rate = rospy.Rate(30) # 30Hz
        while not rospy.is_shutdown():
            now = time.time()
            dt = now - self.last_loop_time
            self.last_loop_time = now
            
            s = self.hub.get_snapshot()
            
            # --- 1. Safety Watchdog (센서 끊김 감지) ---
            cam_age = now - (s.cam.timestamp if s.cam else 0)
            lidar_age = now - (s.lidar.timestamp if s.lidar else 0)
            
            # 1.5초 이상 둘 다 끊기면 비상 정지 (Hard Stop)
            if cam_age > self.cfg.watchdog_hard_sec and lidar_age > self.cfg.watchdog_hard_sec:
                rospy.logwarn_throttle(1.0, "[Safety] Sensor Signal Lost! EMERGENCY STOP.")
                self._publish(self.cfg.steer_center, 0)
                rate.sleep()
                continue
            
            # --- 2. FSM Update ---
            status = self.mission.update(s)
            
            # --- 3. Control Logic ---
            steer = self.cfg.steer_center
            speed = 0

            if status == "STOP" or status == "FINISH":
                # 정지 상태
                steer = self.cfg.steer_center
                speed = 0
            
            elif status == "PARKING":
                # 주차 모드 (AR 태그 추종)
                speed = 0 # 탐색 중엔 정지 or 초저속
                if s.ar_dist is not None:
                    # AR 태그 각도(rad)를 조향(deg)으로 변환
                    angle_deg = math.degrees(s.ar_angle)
                    # AR 각도(rad)->deg 를 PWM으로 변환: 게인은 실차에서 튜닝 필요
                    if self.cfg.steer_invert:
                        angle_deg = -angle_deg
                    steer = self.cfg.steer_center + int(self.cfg.ar_steer_gain * angle_deg)
                    speed = self.cfg.speed_parking
                else:
                    # 태그 찾는 중 (천천히 직진 or 정지)
                    steer = self.cfg.steer_center
                    speed = self.cfg.speed_min

            else:
                # 일반 주행 (Sensor Fusion & Fallback)
                final_err = 0.0
                is_valid = False
                
                # 신뢰도×신선도 기반 퓨전 (튀는 출력 감소)
                cam_err = s.cam.get_error_norm(self.cfg) if s.cam else 0.0
                lidar_err = s.lidar.get_error_norm(self.cfg) if s.lidar else 0.0

                cam_age = (now - s.cam.timestamp) if s.cam else 1e9
                lidar_age = (now - s.lidar.timestamp) if s.lidar else 1e9

                # Freshness: 0~ttl 구간에서 1 -> 0 선형감소
                cam_f = max(0.0, min(1.0, 1.0 - (cam_age / float(self.cfg.cam_ttl_sec)))) if float(self.cfg.cam_ttl_sec) > 1e-6 else 0.0
                lidar_f = max(0.0, min(1.0, 1.0 - (lidar_age / float(self.cfg.lidar_ttl_sec)))) if float(self.cfg.lidar_ttl_sec) > 1e-6 else 0.0

                # Reliability: coeff 개수로 가중치 설정
                cam_n = len(getattr(s.cam, "coeffs", []) or []) if s.cam else 0
                lidar_n = len(getattr(s.lidar, "coeffs", []) or []) if s.lidar else 0

                cam_r = float(self.cfg.cam_double_lane_weight) if cam_n >= 6 else (float(self.cfg.cam_single_lane_weight) if cam_n >= 3 else 0.0)
                lidar_r = float(self.cfg.lidar_double_lane_weight) if lidar_n >= 6 else (float(self.cfg.lidar_single_lane_weight) if lidar_n >= 3 else 0.0)

                w_cam = cam_r * cam_f
                w_lidar = lidar_r * lidar_f

                w_sum = w_cam + w_lidar
                if w_sum > 1e-6:
                    final_err = (w_cam * cam_err + w_lidar * lidar_err) / w_sum
                    is_valid = True
                else:
                    is_valid = False
                if is_valid:
                    steer = self.ctrl.compute_steer(final_err, dt)
                    # 갈림길/단일차선에서 미션 방향으로 약한 바이어스 (기본 0=OFF)
                    if int(self.cfg.fork_bias_pwm) != 0 and (len(getattr(s.cam, 'coeffs', []) or []) == 3) and (len(getattr(s.lidar, 'coeffs', []) or []) < 6):
                        bias = int(self.cfg.fork_bias_pwm)
                        # RIGHT면 +bias, LEFT면 -bias 로 가정. steer_invert면 반전
                        if str(s.mission_direction).upper() == 'LEFT':
                            bias = -bias
                        if self.cfg.steer_invert:
                            bias = -bias
                        steer = max(self.cfg.steer_min, min(self.cfg.steer_max, steer + bias))
                    steer_dev = abs(steer - self.cfg.steer_center)
                    
                    # 곡률에 따른 속도 조절 (코너 감속)
                    if steer_dev > 25:
                        speed = self.cfg.speed_caution
                    else:
                        speed = self.cfg.speed_drive
                else:
                    # Soft Fallback: 0.5초 이내면 관성 주행 (이전 값 유지)
                    if min(cam_age, lidar_age) < self.cfg.watchdog_soft_sec:
                        steer = self.cfg.steer_center - int(self.ctrl.prev_output)
                        speed = self.cfg.speed_caution
                    else:
                        # 너무 오래 끊김 -> 정지
                        speed = 0

            # --- 4. Actuation ---
            steer = self.ctrl._clamp(steer)
            self._publish(steer, speed)
            rate.sleep()

    def _publish(self, steer, speed):
        # 모터 명령
        self.pub_motor.publish(Int16MultiArray(data=[int(steer), int(speed)]))
        # 미션 상태 공유 (Vision 노드 등에서 활용)
        self.pub_dir.publish(self.mission.direction)
        self.pub_status.publish(self.mission._status_str())

if __name__ == "__main__":
    DecisionNode().run()