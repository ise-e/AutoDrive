#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import time

import numpy as np
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Int16MultiArray, String

def _clamp_i(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

class DecisionNode:
    # 상태(백본)만 유지
    S_START = "START"
    S_YELLOW_STOP_1 = "YELLOW_STOP_1"
    S_DRIVE_RIGHT = "DRIVE_RIGHT"
    S_DRIVE_LEFT = "DRIVE_LEFT"
    S_YELLOW_STOP_2 = "YELLOW_STOP_2"
    S_PARKING = "PARKING"
    S_FINISH = "FINISH"

    def __init__(self):
        rospy.init_node("decision_node")

        self._load_params()

        self.state = self.S_START
        self.mission_direction = "RIGHT"
        self.white_count = 0
        self._ts_white = 0.0
        self._ts_yellow = 0.0
        self._ts_ar_missing = 0.0

        self._lock = threading.RLock()
        self._lane = None          # [la,lb,lc, ra,rb,rc] (x(y))
        self._stop = "NONE"        # NONE/WHITE/YELLOW
        self._light = "UNKNOWN"    # UNKNOWN/RED/YELLOW/GREEN
        self._ar = None            # (dist_m, ang_rad)
        self._pf_dir = None        # deg (0~360)
        self._pf_dir_t = 0.0
        self._pf_vel = None        # int speed
        self._pf_vel_t = 0.0

        self.pub_motor = rospy.Publisher("/motor", Int16MultiArray, queue_size=1)
        self.pub_dir = rospy.Publisher("/mission_direction", String, queue_size=1)
        self.pub_status = rospy.Publisher("/mission_status", String, queue_size=1)

        rospy.Subscriber("/lane_coeffs", Float32MultiArray, self._cb_lane, queue_size=1)
        rospy.Subscriber("/stop_line_status", String, lambda m: self._set("_stop", (m.data or "NONE").upper()), queue_size=1)
        rospy.Subscriber("/traffic_light_status", String, lambda m: self._set("_light", (m.data or "UNKNOWN").upper()), queue_size=1)
        rospy.Subscriber("/ar_tag_info", Float32MultiArray, self._cb_ar, queue_size=1)
        rospy.Subscriber("/tunnel/direction", Float32, lambda m: self._set("_pf_dir", float(m.data), "_pf_dir_t"), queue_size=1)
        rospy.Subscriber("/tunnel/velocity", Float32, lambda m: self._set("_pf_vel", float(m.data), "_pf_vel_t"), queue_size=1)

    def _load_params(self):
        p = rospy.get_param
        # 핵심만 남김(기능 유지용)
        defaults = {
            "steer_center": 90,
            "steer_min": 45,
            "steer_max": 135,
            "k_steer": 45.0,
            "speed_drive": 100,
            "speed_stop": 90,
            "speed_parking": 99,
            "speed_min_run": 99,
            "bev_w": 640,
            "bev_h": 480,
            "lane_eval_y": -1.0,
            "white_debounce_sec": 1.0,
            "yellow_debounce_sec": 1.0,
            "pf_enable": True,
            "pf_blend_angle_deg": 20.0,
            "pf_blend_max": 0.6,
            "pf_gain_deg_to_steer": 0.6,
            "pf_timeout_sec": 0.5,
            "park_stop_dist_m": 0.35,
            "park_search_sec": 1.0,
            "park_search_speed": 98,
        }
        for k, d in defaults.items():
            setattr(self, k, p("~" + k, d))

    def _set(self, name, value, t_name=None):
        with self._lock:
            setattr(self, name, value)
            if t_name:
                setattr(self, t_name, time.time())

    def _cb_lane(self, msg):
        arr = list(msg.data)
        if len(arr) >= 6:
            self._set("_lane", arr[:6])

    def _cb_ar(self, msg):
        # format: [id, dist_m, ang_rad]
        d = list(msg.data)
        if len(d) >= 3:
            self._set("_ar", (float(d[1]), float(d[2])))
        else:
            self._set("_ar", None)

    def _snap(self):
        with self._lock:
            return {
                "t": time.time(),
                "lane": self._lane,
                "stop": self._stop,
                "light": self._light,
                "ar": self._ar,
                "pf_dir": self._pf_dir,
                "pf_dir_t": self._pf_dir_t,
                "pf_vel": self._pf_vel,
                "pf_vel_t": self._pf_vel_t,
            }

    def _debounced(self, t_now, t_prev, sec):
        return (t_now - float(t_prev)) >= float(sec)

    def _update_state(self, s):
        prev = self.state
        t = s["t"]
        stop = (s["stop"] or "NONE").upper()
        light = (s["light"] or "UNKNOWN").upper()

        if stop == "WHITE" and self._debounced(t, self._ts_white, self.white_debounce_sec):
            self._ts_white = t
            self.white_count += 1
            if self.white_count >= 2:
                self.mission_direction = "LEFT"

        # 상태 전이(백본 유지)
        if self.state == self.S_START:
            self.state = self.S_YELLOW_STOP_1
            out_status = "NONE"

        elif self.state == self.S_YELLOW_STOP_1:
            if stop == "YELLOW" and self._debounced(t, self._ts_yellow, self.yellow_debounce_sec):
                self._ts_yellow = t
                out_status = "STOP"
            elif stop == "YELLOW":
                if light == "GREEN":
                    self.state = self.S_DRIVE_RIGHT
                    out_status = "NONE"
                else:
                    out_status = "STOP"
            else:
                out_status = "NONE"

        elif self.state == self.S_DRIVE_RIGHT:
            if self.white_count >= 2:
                self.state = self.S_DRIVE_LEFT
            out_status = "NONE"

        elif self.state == self.S_DRIVE_LEFT:
            if stop == "YELLOW" and self._debounced(t, self._ts_yellow, self.yellow_debounce_sec):
                self._ts_yellow = t
                self.state = self.S_YELLOW_STOP_2
                out_status = "STOP"
            else:
                out_status = "NONE"

        elif self.state == self.S_YELLOW_STOP_2:
            if stop == "YELLOW":
                if light == "GREEN":
                    self.state = self.S_PARKING
                    out_status = "PARKING"
                else:
                    out_status = "STOP"
            else:
                out_status = "NONE"

        elif self.state == self.S_PARKING:
            ar = s.get("ar")
            if (ar is not None) and (float(ar[0]) <= float(self.park_stop_dist_m)):
                self.state = self.S_FINISH
                out_status = "FINISH"
            else:
                out_status = "PARKING"

        else:
            out_status = "FINISH"

        if prev != self.state:
            rospy.loginfo("[FSM] %s -> %s | stop=%s light=%s white=%d dir=%s status=%s",
                          prev, self.state, stop, light, int(self.white_count), self.mission_direction, out_status)
        return self.mission_direction, out_status

    def _calc_cmd(self, status, s):
        cen = int(self.steer_center)
        if status in ("STOP", "FINISH"):
            return cen, int(self.speed_stop)

        if status == "PARKING":
            ar = s.get("ar")
            if ar is None:
                if self._ts_ar_missing <= 0.0:
                    self._ts_ar_missing = s["t"]
                if (s["t"] - self._ts_ar_missing) <= float(self.park_search_sec):
                    spd = max(int(self.speed_min_run), int(self.park_search_speed))
                    return cen, spd
                return cen, int(self.speed_stop)

            self._ts_ar_missing = 0.0
            dist_m, ang_rad = float(ar[0]), float(ar[1])
            if dist_m <= float(self.park_stop_dist_m):
                return cen, int(self.speed_stop)

            steer = _clamp_i(int(cen + math.degrees(ang_rad)), int(self.steer_min), int(self.steer_max))
            spd = max(int(self.speed_min_run), int(self.speed_parking))
            return steer, spd

        # DRIVE
        steer = cen
        speed = int(self.speed_drive)

        coeffs = s.get("lane")
        if coeffs:
            y = (float(self.bev_h) - 1.0) if float(self.lane_eval_y) < 0.0 else float(self.lane_eval_y)
            xl = np.polyval(coeffs[0:3], y)
            xr = np.polyval(coeffs[3:6], y)
            cx = 0.5 * (xl + xr)
            half = max(1.0, 0.5 * float(self.bev_w))
            err = (half - cx) / half
            steer = _clamp_i(int(cen + err * float(self.k_steer)), int(self.steer_min), int(self.steer_max))

        if bool(self.pf_enable) and (s.get("pf_dir") is not None):
            if (s["t"] - float(s.get("pf_dir_t", 0.0))) <= float(self.pf_timeout_sec):
                d = float(s["pf_dir"]); d = d if d <= 180.0 else (d - 360.0)
                pf_steer = float(cen) - d * float(self.pf_gain_deg_to_steer)
                w = min(abs(d) / max(float(self.pf_blend_angle_deg), 1e-6), 1.0) * float(self.pf_blend_max)
                steer = _clamp_i(int((1.0 - w) * float(steer) + w * float(pf_steer)),
                                 int(self.steer_min), int(self.steer_max))
                if s.get("pf_vel") is not None and (s["t"] - float(s.get("pf_vel_t", 0.0))) <= float(self.pf_timeout_sec):
                    speed = min(int(speed), int(s["pf_vel"]))

        return int(steer), max(int(speed), int(self.speed_min_run))

    def _publish(self, steer, speed, mdir, mstatus):
        self.pub_motor.publish(Int16MultiArray(data=[int(steer), int(speed)]))
        self.pub_dir.publish(String(data=str(mdir)))
        self.pub_status.publish(String(data=str(mstatus)))

    def run(self):
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            s = self._snap()
            mdir, mstatus = self._update_state(s)
            steer, speed = self._calc_cmd(mstatus, s)
            self._publish(steer, speed, mdir, mstatus)
            r.sleep()

if __name__ == "__main__":
    DecisionNode().run()
