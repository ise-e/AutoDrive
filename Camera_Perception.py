#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera_Perception_refactored.py (기능 유지 + 직관성 개선)

토픽 계약(간단 요약)
- 입력:
  - ~camera_topic (sensor_msgs/Image) default: /camera/image_raw
  - /mission_direction (String) : "LEFT" | "RIGHT"
  - /mission_status (String) : "NONE" | "STOP" | "PARKING" | ...

- 출력:
  - /lane_coeffs (Float32MultiArray) : [la, lb, lc, ra, rb, rc]
      x_px = a*y_px^2 + b*y_px + c  (BEV 이미지 좌표계)
  - /stop_line_status (String) : "NONE" | "WHITE" | "YELLOW"
  - /traffic_light_status (String) : "RED" | "YELLOW" | "GREEN" | "UNKNOWN"
  - /ar_tag_info (Float32MultiArray) : [tag_id, dist_m, ang_rad, ...]
  - (옵션) /debug/lane_overlay/image/compressed (CompressedImage) : 오버레이 JPG

핵심 개선
- lambda setattr 제거 -> 명시적 콜백 메서드로 추적성 향상
- gp/sp/dp 같은 축약어 제거 -> get_param, src_pts_ratio/dst_pts_ratio 등으로 명확화
"""

from __future__ import annotations

import time
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, String


def now_sec() -> float:
    return time.time()


class CameraPerception:
    def __init__(self):
        rospy.init_node("camera_perception_node")
        get_param = rospy.get_param

        # ---------- IO ----------
        self.cam_topic = get_param("~camera_topic", "/camera/image_raw")
        self.w, self.h = int(get_param("~out_w", 640)), int(get_param("~out_h", 480))

        self.overlay_topic = get_param("~overlay_topic", "/debug/lane_overlay/image/compressed")
        self.overlay_on = bool(get_param("~overlay_enable", True))
        self.jpg_q = int(get_param("~jpeg_quality", 80))

        # ---------- BEV ----------
        src_pts_ratio = get_param(
            "~src_pts_ratio",
            [200/640, 300/480, 440/640, 300/480, 50/640, 450/480, 590/640, 450/480]
        )
        dst_pts_ratio = get_param(
            "~dst_pts_ratio",
            [150/640, 0/480, 490/640, 0/480, 150/640, 1.0, 490/640, 1.0]
        )
        self.M = cv2.getPerspectiveTransform(
            self._ratio_pts(src_pts_ratio, self.w, self.h),
            self._ratio_pts(dst_pts_ratio, self.w, self.h),
        )

        # ---------- lane fit ----------
        self.nw = int(get_param("~nwindows", 9))
        self.margin = int(get_param("~margin", 60))
        self.minpix = int(get_param("~minpix", 40))
        self.minpts = int(get_param("~min_points", 220))

        # ---------- stopline ----------
        self.stop_y0 = float(get_param("~stop_check_y0", 0.70))
        self.stop_y1 = float(get_param("~stop_check_y1", 0.80))
        self.stop_thr = float(get_param("~stop_px_per_col", 15.0))
        self.lane_cut = float(get_param("~lane_fit_ymax", 0.70))

        # ---------- dropout hold ----------
        self.hold_sec = float(get_param("~last_fit_hold_sec", 0.5))
        self.last_fit = None
        self.last_fit_t = 0.0

        # ---------- mission state ----------
        self.mdir = "RIGHT"
        self.mstatus = "NONE"

        # ---------- HSV ranges ----------
        self.RED1 = (np.array([0, 100, 100]),   np.array([10, 255, 255]))
        self.RED2 = (np.array([160, 100, 100]), np.array([179, 255, 255]))
        self.YELLOW = (np.array([15, 100, 100]), np.array([35, 255, 255]))
        self.GREEN  = (np.array([45, 100, 100]), np.array([90, 255, 255]))
        self.WHITE  = (np.array([0, 0, 200]),    np.array([179, 40, 255]))

        # ---------- pubs ----------
        self.pub_lane = rospy.Publisher("/lane_coeffs", Float32MultiArray, queue_size=1)
        self.pub_stop = rospy.Publisher("/stop_line_status", String, queue_size=1)
        self.pub_tl   = rospy.Publisher("/traffic_light_status", String, queue_size=1)
        self.pub_ar   = rospy.Publisher("/ar_tag_info", Float32MultiArray, queue_size=1)
        self.pub_ov   = rospy.Publisher(self.overlay_topic, CompressedImage, queue_size=1)

        # ---------- subs ----------
        rospy.Subscriber(self.cam_topic, Image, self._on_img, queue_size=1)
        rospy.Subscriber("/mission_direction", String, self._on_mission_direction, queue_size=1)
        rospy.Subscriber("/mission_status", String, self._on_mission_status, queue_size=1)

        rospy.loginfo("[CamPerc] refactored | cam=%s overlay=%s", self.cam_topic, self.overlay_topic)

    # ---------------- mission callbacks ----------------
    def _on_mission_direction(self, msg: String):
        self.mdir = (msg.data or "RIGHT").upper()

    def _on_mission_status(self, msg: String):
        self.mstatus = (msg.data or "NONE").upper()

    # ---------------- helper ----------------
    @staticmethod
    def _ratio_pts(r, w, h):
        r = [float(x) for x in r]
        r = (r + r[:8])[:8]
        return np.float32([
            [r[0] * w, r[1] * h],
            [r[2] * w, r[3] * h],
            [r[4] * w, r[5] * h],
            [r[6] * w, r[7] * h],
        ])

    @staticmethod
    def _rosimg_to_bgr(msg: Image):
        if msg.height <= 0 or msg.width <= 0:
            return None
        enc = (msg.encoding or "").lower()
        buf = np.frombuffer(msg.data, np.uint8)

        if enc in ("bgr8", "rgb8"):
            img = buf.reshape((msg.height, msg.width, 3))
            return img if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if enc in ("mono8", "8uc1"):
            g = buf.reshape((msg.height, msg.width))
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        return None

    def _publish_overlay(self, bgr):
        if not self.overlay_on:
            return
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_q])
        if not ok:
            return
        m = CompressedImage()
        m.header.stamp = rospy.Time.now()
        m.format = "jpeg"
        m.data = enc.tobytes()
        self.pub_ov.publish(m)

    # ---------------- main callback ----------------
    def _on_img(self, msg: Image):
        frame = self._rosimg_to_bgr(msg)
        if frame is None:
            return

        frame = cv2.resize(frame, (self.w, self.h))
        debug_frame = frame.copy()
        bev = cv2.warpPerspective(frame, self.M, (self.w, self.h))

        lfit, rfit, stop, debug_bev = self._lane_stop(bev)
        self.pub_stop.publish(stop)

        # 양쪽 차선 모두 검지
        if lfit is not None and rfit is not None:
            # 이미지 최하단
            y_bot = self.h - 1
            # 이미지 상단
            y_top = int(self.h * 0.3)
            # 각 지점에서의 너비 계산
            w_bot = self._poly_x(rfit, y_bot) - self._poly_x(lfit, y_bot)
            w_top = self._poly_x(rfit, y_top) - self._poly_x(lfit, y_top)
            # 너비 변화량 (Width Deviation)
            width_deviation = abs(w_top - w_bot)
            # 차이가 60 이상일 경우 **값 조정 필요
            if width_deviation > 60:
                if w_top < w_bot:  # 상단으로 갈수록 너비가 좁아지는 경우 (수렴)
                    if self.mdir == "LEFT":
                        # 왼쪽으로 가야 하는데 왼쪽이 좁아지면, 확실한 오른쪽 선을 기준으로 주행
                        self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
                    else: # RIGHT 미션
                        # 오른쪽으로 가야 하는데 오른쪽이 좁아지면, 왼쪽 선을 기준으로 주행
                        self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
                else:  # 상단으로 갈수록 너비가 넓어지는 경우 (발산/갈림길)
                    if self.mdir == "LEFT":
                        # 왼쪽 갈림길을 타기 위해 왼쪽 차선을 가이드로 선택
                        self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
                    else: # RIGHT 미션
                        # 오른쪽 갈림길을 타기 위해 오른쪽 차선을 가이드로 선택
                        self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
            # 정상적인 경우
            else :
                self.last_fit, self.last_fit_t = (lfit, rfit), now_sec()
                self.pub_lane.publish(Float32MultiArray(data=list(lfit) + list(rfit)))
        # 왼쪽만 검지
        elif lfit is not None:
            self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
        # 오른쪽만 검지
        elif rfit is not None:
            self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
        # 둘 다 미검지 ** 이 부분은 나중에 판단 노드로 옮길 생각입니다
        else:
            if self.last_fit and (now_sec() - self.last_fit_t) <= self.hold_sec:
                lf, rf = self.last_fit
                self.pub_lane.publish(Float32MultiArray(data=list(lf) + list(rf)))

        if self.mstatus == "STOP":
            debug_frame = self._traffic_light(frame, debug_frame) or debug_frame
        if self.mstatus == "PARKING":
            debug_frame = self._ar_tag(frame, debug_frame) or debug_frame

        combined_view = np.hstack([debug_frame, debug_bev])
        self._publish_overlay(combined_view)

    # ---------------- algorithms (원본 로직 유지) ----------------
    def _lane_stop(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        ymask = cv2.inRange(hsv, *self.YELLOW)
        wmask = cv2.inRange(hsv, *self.WHITE)

        h, w = ymask.shape
        y0, y1 = int(self.stop_y0 * h), int(self.stop_y1 * h)
        if y1 <= y0:
            y1 = min(h, y0 + 1)

        ypx = cv2.countNonZero(ymask[y0:y1, :])
        wpx = cv2.countNonZero(wmask[y0:y1, :])

        stop = "NONE"
        if ypx > w * self.stop_thr:
            stop = "YELLOW"
        elif wpx > w * self.stop_thr:
            stop = "WHITE"

        lane = cv2.bitwise_or(ymask, wmask)
        cut = int(self.lane_cut * h)
        lane[cut:, :] = 0
        lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        lfit, rfit = self._fit(lane)
        debug_bev = self._debug_line(bev, lane, lfit, rfit, stop, ypx, wpx, stop)
        return lfit, rfit, stop, debug_bev

    @staticmethod
    def _base(hist, lo, hi, fallback):
        lo, hi = max(0, int(lo)), min(len(hist), int(hi))
        if hi <= lo:
            return int(fallback)
        seg = hist[lo:hi]
        if np.max(seg) <= 0:
            return int(fallback)
        return int(np.argmax(seg) + lo)

    def _fit(self, binimg):
        hist = np.sum(binimg[binimg.shape[0] // 2:, :], axis=0)
        mid = len(hist) // 2

        if self.mdir == "RIGHT":
            lx = self._base(hist, mid - 100, mid + 100, mid // 2)
            rx = self._base(hist, mid + 150, len(hist), mid + mid // 2)
        elif self.mdir == "LEFT":
            lx = self._base(hist, 0, mid - 150, mid // 2)
            rx = self._base(hist, mid - 100, mid + 100, mid + mid // 2)
        else:
            lx = self._base(hist, 0, mid, mid // 2)
            rx = self._base(hist, mid, len(hist), mid + mid // 2)

        nw = max(3, self.nw)
        win_h = binimg.shape[0] // nw
        nz = binimg.nonzero()
        nzy, nzx = np.array(nz[0]), np.array(nz[1])
        lidx, ridx = [], []

        for win in range(nw):
            ylo = binimg.shape[0] - (win + 1) * win_h
            yhi = binimg.shape[0] - win * win_h
            llo, lhi = lx - self.margin, lx + self.margin
            rlo, rhi = rx - self.margin, rx + self.margin

            gl = ((nzy >= ylo) & (nzy < yhi) & (nzx >= llo) & (nzx < lhi)).nonzero()[0]
            gr = ((nzy >= ylo) & (nzy < yhi) & (nzx >= rlo) & (nzx < rhi)).nonzero()[0]
            lidx.append(gl)
            ridx.append(gr)
            
            if len(gl) > self.minpix:
                lx = int(np.mean(nzx[gl]))
            if len(gr) > self.minpix:
                rx = int(np.mean(nzx[gr]))

        l_fit_res, r_fit_res = None, None
        try:
            li = np.concatenate(lidx) if lidx else np.array([], np.int32)
            if len(li) >= self.minpts:
                l_fit_res = np.polyfit(nzy[li], nzx[li], 2)
            
            ri = np.concatenate(ridx) if ridx else np.array([], np.int32)
            if len(ri) >= self.minpts:
                r_fit_res = np.polyfit(nzy[ri], nzx[ri], 2)
        except Exception:
            pass

        return l_fit_res, r_fit_res

    @staticmethod
    def _poly_x(f, y):
        return (f[0] * y * y) + (f[1] * y) + f[2]

    def _debug_line(self, bev, lane, lf, rf, stop, ypx, wpx, color):
        debug_bev = bev.copy()
        debug_bev[:, :, 1] = np.maximum(debug_bev[:, :, 1], (lane // 2).astype(np.uint8))

        # ===== [DEBUG] 시각화 코드 ========
        if color != "NONE":
            h, w = debug_bev.shape[:2]
            y0 = int(self.stop_y0 * h)
            y1 = int(self.stop_y1 * h)

            box_color = (0, 255, 255) if stop == "YELLOW" else (200, 200, 200)

            cv2.rectangle(debug_bev, (0, y0), (w, y1), box_color, 4)

            cv2.putText(debug_bev, f"{stop} LINE", (w//2 - 100, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        # ================================

        ok = (lf is not None and rf is not None)
        if ok:
            ptsL, ptsR, ptsC = [], [], []
            h, w = debug_bev.shape[:2]
            for y in range(h - 1, 0, -20):
                lx, rx = self._poly_x(lf, y), self._poly_x(rf, y)
                if 0 <= lx < w and 0 <= rx < w:
                    cx = 0.5 * (lx + rx)
                    ptsL.append((int(lx), y))
                    ptsR.append((int(rx), y))
                    ptsC.append((int(cx), y))
            if len(ptsC) >= 2:
                cv2.polylines(debug_bev, [np.array(ptsL)], False, (255, 0, 0), 2)
                cv2.polylines(debug_bev, [np.array(ptsR)], False, (0, 0, 255), 2)
                cv2.polylines(debug_bev, [np.array(ptsC)], False, (80, 255, 80), 2)

        cv2.putText(debug_bev, f"{self.mdir} {self.mstatus} STOP:{stop} FIT:{'OK' if ok else 'NO'}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(debug_bev, f"Y:{ypx} W:{wpx}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        return debug_bev

    # 신호등 탐지 시각화 함수
    def _debug_light(self, debug_frame, min_h, max_h, min_w, max_w, light):
        color = (0, 0, 0)
        if light == "GREEN":
            color = (0, 255, 0)
        elif light == "YELLOW":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(debug_frame, (min_w, min_h), (max_w, max_h) , color, thickness=3)

    def _traffic_light(self, frame, debug_frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        roi = hsv[0:int(h * 0.4), int(w * 0.2):int(w * 0.8)]
        red = cv2.countNonZero(cv2.bitwise_or(cv2.inRange(roi, *self.RED1), cv2.inRange(roi, *self.RED2)))
        yel = cv2.countNonZero(cv2.inRange(roi, *self.YELLOW))
        grn = cv2.countNonZero(cv2.inRange(roi, *self.GREEN))
        best = max((red, "RED"), (yel, "YELLOW"), (grn, "GREEN"))[1]
        light_result = best if max(red, yel, grn) > 200 else "UNKNOWN"
        self.pub_tl.publish(light_result)

        # ==============DEBUG==============
        if light_result != "UNKNOWN":
            self._debug_light(debug_frame, 0, int(h * 0.4), int(w * 0.2), int(w * 0.8), light_result)
        return debug_frame

    def _ar_tag(self, frame, debug_frame):
        if not hasattr(cv2, "aruco"):
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ar = cv2.aruco
        dic = ar.Dictionary_get(ar.DICT_6X6_250)
        prm = ar.DetectorParameters_create()
        corners, ids, _ = ar.detectMarkers(gray, dic, parameters=prm)
        if ids is None:
            return

        cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids, (255, 0, 0)) # 시각화 코드

        tag = float(rospy.get_param("~tag_size", 0.15))
        focal = float(rospy.get_param("~focal_length", 600.0))
        cx = gray.shape[1] * 0.5

        out = []
        for i in range(len(ids)):
            c = corners[i][0]
            pw = float(np.linalg.norm(c[0] - c[1]))
            if pw <= 1e-6:
                continue
            dist = (tag * focal) / pw
            ang = float(np.arctan2(float(np.mean(c[:, 0]) - cx), focal))
            out += [float(ids[i]), dist, ang]
        if out:
            self.pub_ar.publish(Float32MultiArray(data=out))
        return debug_frame

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    CameraPerception().run()
