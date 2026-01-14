#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DecisionNode
- 개선된 I/O / PF 블렌드 / Lane RViz marker 구조는 유지
- 상태머신만 "기존 대회 시나리오 FSM"으로 사용
- ✅ FSM 로그 추가:
  1) 상태 전이 로그: 전이 발생 시 1회 출력
  2) 상태 유지 로그: 1Hz throttle로 현재 입력/출력/명령 출력
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import rospy
from std_msgs.msg import Int16MultiArray, String, Float32MultiArray, Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# ---------------- Data Structures ----------------
@dataclass
class Lane:
    """차선 다항식 계수: [la, lb, lc, ra, rb, rc]"""
    c: List[float]

    def x_center(self, y: float) -> float:
        la, lb, lc, ra, rb, rc = self.c
        lx = la * y * y + lb * y + lc
        rx = ra * y * y + rb * y + rc
        return 0.5 * (lx + rx)

    def x_poly(self, y: float, is_left: bool) -> float:
        base = 0 if is_left else 3
        return self.c[base] * y * y + self.c[base + 1] * y + self.c[base + 2]


@dataclass
class Snap:
    """루프 1회에서 사용하는 입력 스냅샷(원자적 캡처)"""
    t: float
    lane: Optional[Lane]
    obs_lane: Optional[Lane]
    stop: str
    light: str
    ar: Optional[Tuple[float, float]]   # (dist_m, ang_rad)

@dataclass
class FsmOut:
    """FSM 출력(카메라 노드와의 계약)"""
    mission_direction: str   # "RIGHT" | "LEFT"
    mission_status: str      # "NONE" | "STOP" | "PARKING" | "FINISH"


# ---------------- 기존 시나리오 FSM ----------------
class LegacyScenarioFSM:
    """
    "기존 Decision(대회 시나리오) 상태머신" 구현 + 로그 포함

    시나리오 요약:
    - START -> YELLOW_STOP_1 -> DRIVE_RIGHT -> DRIVE_LEFT -> YELLOW_STOP_2 -> PARKING -> FINISH
    - WHITE 2회 카운트 -> mission_direction = LEFT
    - 1차 YELLOW: STOP 유지, GREEN이면 DRIVE_RIGHT로 전환
    - 2차 YELLOW: STOP 유지, GREEN이면 PARKING으로 전환
    - PARKING: AR dist <= park_stop_dist_m 이면 FINISH
    """

    START = "START"
    YELLOW_STOP_1 = "YELLOW_STOP_1"
    DRIVE_RIGHT = "DRIVE_RIGHT"
    DRIVE_LEFT = "DRIVE_LEFT"
    YELLOW_STOP_2 = "YELLOW_STOP_2"
    PARKING = "PARKING"
    FINISH = "FINISH"

    def __init__(self, stopline_debounce_sec: float, park_stop_dist_m: float):
        self.state = self.START

        self.stopline_debounce_sec = float(stopline_debounce_sec)
        self.park_stop_dist_m = float(park_stop_dist_m)

        self.white_count = 0
        self.mission_direction = "RIGHT"
        self.is_stop = False

        self._ts_white = 0.0
        self._ts_yellow = 0.0

        self._state_name = {
            self.START: "START",
            self.YELLOW_STOP_1: "YELLOW_STOP_1",
            self.DRIVE_RIGHT: "DRIVE_RIGHT",
            self.DRIVE_LEFT: "DRIVE_LEFT",
            self.YELLOW_STOP_2: "YELLOW_STOP_2",
            self.PARKING: "PARKING",
            self.FINISH: "FINISH",
        }

    def _debounced(self, now_t: float, last_t: float) -> bool:
        return (now_t - last_t) >= self.stopline_debounce_sec

    def _log_transition(self, prev_state: str, new_state: str, s: Snap, out: FsmOut) -> None:
        rospy.loginfo(
            "[FSM] %s -> %s | stop=%s light=%s white=%d ar=%s | dir=%s status=%s",
            self._state_name.get(prev_state, prev_state),
            self._state_name.get(new_state, new_state),
            (s.stop or "NONE"),
            (s.light or "UNKNOWN"),
            self.white_count,
            ("None" if s.ar is None else f"{s.ar[0]:.2f}m,{math.degrees(s.ar[1]):.1f}deg"),
            out.mission_direction,
            out.mission_status,
        )

    def step(self, s: Snap) -> FsmOut:
        prev_state = self.state

        stop = (s.stop or "NONE").upper()
        light = (s.light or "UNKNOWN").upper()

        # --- WHITE 카운트(디바운스) ---
        if stop == "WHITE" and self._debounced(s.t, self._ts_white):
            self._ts_white = s.t
            self.white_count += 1
            rospy.loginfo("[FSM] WHITE++ -> %d", self.white_count)
            if self.white_count >= 2:
                self.mission_direction = "LEFT"

        # --- 상태 전이 ---
        if self.state == self.START:
            # 시작 직후 1차 노란선 정지 상태로 진입
            self.state = self.YELLOW_STOP_1
            out = FsmOut(self.mission_direction, "NONE")
            if prev_state != self.state:
                self._log_transition(prev_state, self.state, s, out)
            return out

        if self.state == self.YELLOW_STOP_1:
            # 노란선 감지: STOP, GREEN이면 출발
            if not self.is_stop and stop == "YELLOW":
                self.is_stop = True
                return FsmOut(self.mission_direction, "STOP")
            elif light == "GREEN":
                self.state = self.DRIVE_RIGHT
                self.is_stop = False
                out = FsmOut(self.mission_direction, "NONE")
                if prev_state != self.state:
                    self._log_transition(prev_state, self.state, s, out)
                return out
            elif light == "UNKNOWN":
                self.is_stop = False
                return FsmOut(self.mission_direction, "NONE")
            else:
                self.is_stop = True
                return FsmOut(self.mission_direction, "STOP")

        if self.state == self.DRIVE_RIGHT:
            # WHITE 2회면 LEFT 주행으로
            if self.white_count >= 2:
                self.state = self.DRIVE_LEFT
                out = FsmOut(self.mission_direction, "NONE")
                if prev_state != self.state:
                    self._log_transition(prev_state, self.state, s, out)
                return out
            return FsmOut(self.mission_direction, "NONE")

        if self.state == self.DRIVE_LEFT:
            # 2차 노란선 만나면 STOP2 진입
            if stop == "YELLOW" and self._debounced(s.t, self._ts_yellow):
                self._ts_yellow = s.t
                self.state = self.YELLOW_STOP_2
                out = FsmOut(self.mission_direction, "STOP")
                if prev_state != self.state:
                    self._log_transition(prev_state, self.state, s, out)
                return out
            return FsmOut(self.mission_direction, "NONE")

        if self.state == self.YELLOW_STOP_2:
            # 노란선 위에서는 STOP, GREEN이면 PARKING
            if not self.is_stop and stop == "YELLOW":
                self.is_stop = True
                return FsmOut(self.mission_direction, "STOP")
            elif light == "GREEN":
                self.is_stop = False
                if s.ar is not None:
                    self.state = self.PARKING
                    return FsmOut(self.mission_direction, "PARKING")
                out = FsmOut(self.mission_direction, "NONE")
                if prev_state != self.state:
                    self._log_transition(prev_state, self.state, s, out)
                return out
            elif light == "UNKNOWN":
                self.is_stop = False
                return FsmOut(self.mission_direction, "NONE")
            else:
                self.is_stop = True
                return FsmOut(self.mission_direction, "STOP")

        if self.state == self.PARKING:
            # AR로 접근해서 dist <= park_stop_dist_m면 FINISH
            if s.ar is not None:
                dist_m = float(s.ar[0])
                if dist_m <= self.park_stop_dist_m:
                    self.state = self.FINISH
                    out = FsmOut(self.mission_direction, "FINISH")
                    if prev_state != self.state:
                        self._log_transition(prev_state, self.state, s, out)
                    return out
            return FsmOut(self.mission_direction, "PARKING")

        # FINISH
        return FsmOut(self.mission_direction, "FINISH")


# ---------------- Decision Node ----------------
class DecisionNode:
    def __init__(self):
        rospy.init_node("decision_node")

        gp = lambda n, d: rospy.get_param("~" + n, d)

        self.accum_error = 0
        self.prev_error = 0
        self.prev_steer = None
        
        # --- config ---
        self.cfg = type("Cfg", (), {
            # steering / speed
            "cen": int(gp("steer_center", 90)),
            "min": int(gp("steer_min", 48)), ## stable steer
            "max": int(gp("steer_max", 132)),## stable steer
            "kp": float(gp("kp_steer", 200.0)),  ## ki, kd를 0으로 설정하고 $kp만 올립니다. 차가 라인을 따라가지만 좌우로 흔들리기 시작하는 지점(임계점)을 찾습니다.
            "kd": float(gp("kd_steer", 750)),     ## kd를 조금씩 올립니다. 차가 흔들리는 현상이 줄어들고 부드럽게 라인 중앙으로 복귀하는지 확인합니다. (보통 kd가 주행 안정성에 가장 큰 영향을 줍니다.)
            "ki": float(gp("ki_steer", 0.0001)),     ## 직선 주행 시 차가 중앙에 있지 않고 한쪽으로 쏠린다면 ki를 아주 미세하게 추가합니다. (대부분의 고속 주행에서는 생략 가능합니다.)
            "spd_drive": int(gp("speed_drive", 100)),
            "spd_stop": int(gp("speed_stop", 90)),
            "spd_parking": int(gp("speed_parking", 99)),

            # run-speed floor (90~97은 정지로 해석되는 플랫폼 대응)
            "speed_min_run": int(gp("speed_min_run", 99)),

            # BEV geometry
            "w": int(gp("bev_w", 640)),
            "h": int(gp("bev_h", 480)),

            # stopline / parking
            "t_db": float(gp("stopline_debounce_sec", 1.0)),
            "dist_p": float(gp("park_stop_dist_m", 0.3)),

            # lane eval
            "lane_eval_y": float(gp("lane_eval_y", -1.0)),
            "obs_eval_x": float(gp("obs_eval_x", 0.5)),

            # parking search (AR 없을 때)
            "park_search_sec": float(gp("park_search_sec", 1.0)),
            "park_search_speed": int(gp("park_search_speed", 98)),

            # viz
            "base_frame": str(gp("base_frame", "base_link")),
            "meters_per_pixel_x": float(gp("meters_per_pixel_x", 9.0703e-4)), # w의 3분의 1일 때
            "meters_per_pixel_y": float(gp("meters_per_pixel_y", 0.01)),
            "lane_marker_enable": bool(gp("lane_marker_enable", True)),
            "lane_marker_topic": str(gp("lane_marker_topic", "/viz/lanes")),
            "lane_marker_width": float(gp("lane_marker_width", 0.02)),
            "lane_marker_lifetime": float(gp("lane_marker_lifetime", 0.2)),
            "lane_sample_step_px": int(gp("lane_sample_step_px", 40)),
        })()

        # --- FSM ---
        self.fsm = LegacyScenarioFSM(
            stopline_debounce_sec=self.cfg.t_db,
            park_stop_dist_m=self.cfg.dist_p
        )

        # --- input cache ---
        self.lock = threading.RLock()
        self._d: Dict[str, Any] = {
            "lane": None,
            "obs_lane": None,
            "stop": "NONE",
            "light": "UNKNOWN",
            "ar": None,
        }

        # --- publishers ---
        self.pub_motor = rospy.Publisher("/motor", Int16MultiArray, queue_size=1)
        self.pub_dir = rospy.Publisher("/mission_direction", String, queue_size=1)
        self.pub_status = rospy.Publisher("/mission_status", String, queue_size=1)

        self.pub_viz = None
        if self.cfg.lane_marker_enable and self.cfg.lane_marker_topic:
            self.pub_viz = rospy.Publisher(self.cfg.lane_marker_topic, MarkerArray, queue_size=1)

        # --- subscribers ---
        rospy.Subscriber("/lane_coeffs", Float32MultiArray, self._cb_lane, queue_size=1)
        rospy.Subscriber("/obs_lane_coeffs", Float32MultiArray, self._cb_obs_lane, queue_size=1)
        rospy.Subscriber("/stop_line_status", String, lambda m: self._up("stop", (m.data or "NONE").upper()), queue_size=1)
        rospy.Subscriber("/traffic_light_status", String, lambda m: self._up("light", (m.data or "UNKNOWN").upper()), queue_size=1)
        rospy.Subscriber("/ar_tag_info", Float32MultiArray, self._cb_ar, queue_size=1)

        self._ts_ar_missing = 0.0

        rospy.loginfo("[Decision] Legacy FSM + FSM logs enabled.")

    # ---------------- Cache update ----------------
    def _up(self, key: str, val: Any, tkey: Optional[str] = None) -> None:
        now = rospy.get_time()
        with self.lock:
            self._d[key] = val
            if tkey:
                self._d[tkey] = now

    def _cb_lane(self, m: Float32MultiArray) -> None:
        data = list(m.data) if m.data else []
        n = len(data)
        y_eval = 470
        LINE_WIDTH_PX = 400  # 실제 BEV상 차선 간격 px값에 맞춰 조정 필요
        if n >= 6:
            # 양쪽 차선 정상 수신
            lane = Lane(data)
        elif n >= 3:
            # 한쪽 차선만 수신 (a, b, c 추출)
            a, b, c = data[0], data[1], data[2]
            target_x = a * (y_eval**2) + b * y_eval + c
            if target_x > 320: # 감지된 것이 오른쪽 차선일 때
                lane = Lane([a, b, c - LINE_WIDTH_PX, a, b, c])
            else: # 감지된 것이 왼쪽 차선일 때
                lane = Lane([a, b, c, a, b, c + LINE_WIDTH_PX])
        else:
            lane = None
        self._up("lane", lane)

    def _cb_obs_lane(self, m: Float32MultiArray) -> None:
        data = list(m.data) if len(m.data) else []
        if len(data):
            obs_lane = Lane(data)
        else:
            obs_lane = None
        self._up("obs_lane", obs_lane)

    def _cb_ar(self, m: Float32MultiArray) -> None:
        # format: [id, dist, ang(rad)]
        ar = (float(m.data[1]), float(m.data[2])) if (m.data and len(m.data) >= 3) else None
        self._up("ar", (ar[0], ar[1]) if ar else None)

    def _capture(self) -> Snap:
        t = rospy.get_time()
        with self.lock:
            return Snap(
                t=t,
                lane=self._d["lane"],
                obs_lane=self._d["obs_lane"],
                stop=self._d["stop"],
                light=self._d["light"],
                ar=self._d["ar"],
            )

    # ---------------- Main loop ----------------
    def run(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            s = self._capture()

            f = self.fsm.step(s)

            # compute cmd
            if f.mission_status == "PARKING":
                steer, speed = self._parking_cmd(s)
            elif f.mission_status in ("STOP", "FINISH"):
                steer, speed = self.cfg.cen, self.cfg.spd_stop
            else:
                steer, speed = self._drive_cmd(s)

            # ✅ 상태 유지 로그 (1Hz)
            rospy.loginfo_throttle(
                1.0,
                "[FSM] state=%s white=%d stop=%s light=%s ar=%s | dir=%s status=%s | cmd=(%d,%d)",
                self.fsm.state,
                self.fsm.white_count,
                (s.stop or "NONE"),
                (s.light or "UNKNOWN"),
                ("None" if s.ar is None else f"{s.ar[0]:.2f}m,{math.degrees(s.ar[1]):.1f}deg"),
                f.mission_direction,
                f.mission_status,
                int(steer), int(speed),
            )

            self._publish(steer, speed, s, f)
            rate.sleep()

    # ---------------- Command computation ----------------
    def _drive_cmd(self, s: Snap) -> Tuple[int, int]:
        steer = int(self.cfg.cen)
        speed = int(self.cfg.spd_drive)

        # 1) Lane -> steer
        if s.obs_lane:
            speed = 98
            err_norm = s.obs_lane.x_center(self.cfg.obs_eval_x)
        elif s.lane:
            y = int(self.cfg.h * 0.8) if (self.cfg.lane_eval_y < 0) else float(self.cfg.lane_eval_y)
            half_w = max(1.0, float(self.cfg.w)/2.0)
            err_norm = (half_w - s.lane.x_center(y)) * self.cfg.meters_per_pixel_x
            THRESHOLD = 400
            if abs(err_norm/self.cfg.meters_per_pixel_x - self.prev_error/self.cfg.meters_per_pixel_x) > THRESHOLD: err_norm = self.prev_error
        else:
            self.accum_error = 0.0
            self.prev_error = 0.0
            return int(self.cfg.cen), max(int(speed), int(self.cfg.speed_min_run))

        self.accum_error += err_norm
        p_term = err_norm * float(self.cfg.kp)
        d_term = (err_norm - self.prev_error) * float(self.cfg.kd)
        i_term = self.accum_error * float(self.cfg.ki)
        steer = self.cfg.cen - int(p_term + d_term + i_term)
        self.prev_error = err_norm
        steer = self._clamp_i(steer, self.cfg.min, self.cfg.max)

        target_steer = self.cfg.cen - int(p_term + d_term + i_term)

        if self.prev_steer:
            alpha = 0.0  ## 현재 kd와 기능이 겹쳐 비활성화, 추후 제거
            steer = int(alpha * self.prev_steer + (1 - alpha) * target_steer)

        self.prev_steer = steer
        steer = self._clamp_i(steer, self.cfg.min, self.cfg.max)

        # 90~97은 정지로 해석되는 경우가 있어, 주행 중에는 최소 속도를 보장
        speed = max(int(speed), int(self.cfg.speed_min_run))
        return int(steer), int(speed)

    def _parking_cmd(self, s: Snap) -> Tuple[int, int]:
        # AR 없으면 잠깐 천천히 탐색 -> 그래도 없으면 정지
        if s.ar is None:
            if self._ts_ar_missing <= 0.0:
                self._ts_ar_missing = s.t
            if (s.t - self._ts_ar_missing) <= float(self.cfg.park_search_sec):
                return int(self.cfg.cen), max(int(self.cfg.speed_min_run), int(self.cfg.park_search_speed))
            return int(self.cfg.cen), int(self.cfg.spd_stop)

        self._ts_ar_missing = 0.0
        dist_m, ang_rad = float(s.ar[0]), float(s.ar[1])

        if dist_m <= float(self.cfg.dist_p):
            return int(self.cfg.cen), int(self.cfg.spd_stop)

        steer = int(self.cfg.cen + int(math.degrees(ang_rad)))
        steer = self._clamp_i(steer, self.cfg.min, self.cfg.max)
        return int(steer), max(int(self.cfg.speed_min_run), int(self.cfg.spd_parking))

    # ---------------- Publish ----------------
    def _publish(self, steer: int, speed: int, s: Snap, f: FsmOut) -> None:
        steer = self._clamp_i(int(steer), self.cfg.min, self.cfg.max)
        speed = int(speed)

        # motor
        self.pub_motor.publish(Int16MultiArray(data=[steer, speed]))

        # mission topics (Camera_Perception 연동)
        self.pub_dir.publish(f.mission_direction)
        # FINISH는 외부에서 STOP처럼 다루는 경우가 많아 STOP으로 내보냄
        self.pub_status.publish("STOP" if f.mission_status == "FINISH" else f.mission_status)

        # viz
        if self.pub_viz and s.lane:
            try:
                self._viz_lane(s.lane)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[Decision] lane marker publish failed: {e}")

    # ---------------- Viz ----------------
    def _px_to_base(self, x_px: float, y_px: float) -> Tuple[float, float]:
        cx = float(self.cfg.w) * 0.5
        forward_m = (float(self.cfg.h) - float(y_px)) * float(self.cfg.meters_per_pixel_y)
        left_m = -(float(x_px) - cx) * float(self.cfg.meters_per_pixel_x)
        return float(forward_m), float(left_m)

    def _line_marker(self, ns: str, mid: int, pts_xy, rgb) -> Marker:
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.cfg.base_frame
        m.ns, m.id = ns, int(mid)
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(self.cfg.lane_marker_width)
        m.color.r, m.color.g, m.color.b, m.color.a = float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.9
        m.lifetime = rospy.Duration(float(self.cfg.lane_marker_lifetime))
        for x, y in pts_xy:
            m.points.append(Point(x=float(x), y=float(y), z=0.05))
        return m

    def _viz_lane(self, lane: Lane) -> None:
        step = max(1, int(self.cfg.lane_sample_step_px))
        left_pts, right_pts, center_pts = [], [], []

        for y in range(int(self.cfg.h) - 1, -1, -step):
            y_f = float(y)
            lx = lane.x_poly(y_f, True)
            rx = lane.x_poly(y_f, False)
            cx = 0.5 * (lx + rx)
            left_pts.append(self._px_to_base(lx, y_f))
            right_pts.append(self._px_to_base(rx, y_f))
            center_pts.append(self._px_to_base(cx, y_f))

        if len(center_pts) < 2:
            return

        ma = MarkerArray()
        ma.markers = [
            self._line_marker("lane_left", 0, left_pts, (0.2, 0.6, 1.0)),
            self._line_marker("lane_right", 1, right_pts, (1.0, 0.6, 0.2)),
            self._line_marker("lane_center", 2, center_pts, (0.3, 1.0, 0.3)),
        ]
        self.pub_viz.publish(ma)

    # ---------------- utils ----------------
    @staticmethod
    def _clamp_i(v: int, low: int, high: int) -> int:
        return max(int(low), min(int(high), int(v)))


if __name__ == "__main__":
    DecisionNode().run()
