#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera_Perception_refactored.py
- ✅ [수정] Center Path Subscriber 콜백 오류 수정 및 시각화 추가
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

        # [수정] center path 저장 변수
        self.center_path_pts = []

        # ---------- stopline ----------
        self.stop_y0 = float(get_param("~stop_check_y0", 0.70))
        self.stop_y1 = float(get_param("~stop_check_y1", 0.80))
        self.stop_thr = float(get_param("~stop_px_per_col", 4.0))
        self.lane_cut = float(get_param("~lane_fit_ymax", 0.70))

        self.stop_line_angle_thr = float(get_param("~stop_line_angle_threshold", 20.0))
        self.stop_line_min_length = float(get_param("~stop_line_min_length", 100.0))

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
        
        # [수정] Center Path 수신용 Subscriber 추가
        rospy.Subscriber("/center", Float32MultiArray, self._on_center_path, queue_size=1)

        rospy.loginfo("[CamPerc] refactored | cam=%s overlay=%s", self.cam_topic, self.overlay_topic)

    # ---------------- mission callbacks ----------------
    def _on_mission_direction(self, msg: String):
        self.mdir = (msg.data or "RIGHT").upper()

    def _on_mission_status(self, msg: String):
        self.mstatus = (msg.data or "NONE").upper()

    # [수정] Center Path 수신 콜백
    def _on_center_path(self, msg: Float32MultiArray):
        data = msg.data
        if not data:
            self.center_path_pts = []
            return
        
        # Decision에서 [x, y, x, y ...] 순서의 Flat List로 옴
        pts = []
        for i in range(0, len(data), 2):
            x = int(data[i])
            y = int(data[i+1])
            pts.append([x, y])
        self.center_path_pts = np.array(pts, dtype=np.int32)

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
            y_bot = self.h - 1
            y_top = int(self.h * 0.3)
            w_bot = self._poly_x(rfit, y_bot) - self._poly_x(lfit, y_bot)
            w_top = self._poly_x(rfit, y_top) - self._poly_x(lfit, y_top)
            width_deviation = abs(w_top - w_bot)
            
            if width_deviation > 60 or w_top < 60:
                if w_top < w_bot:  # 수렴
                    if self.mdir == "LEFT":
                        self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
                    else: 
                        self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
                else:  # 발산
                    if self.mdir == "LEFT":
                        self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
                    else: 
                        self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
            else :
                self.last_fit, self.last_fit_t = (lfit, rfit), now_sec()
                self.pub_lane.publish(Float32MultiArray(data=list(lfit) + list(rfit)))
        elif lfit is not None:
            self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
        elif rfit is not None:
            self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
        else:
            if self.last_fit is not None and (now_sec() - self.last_fit_t) < self.hold_sec:
                lfit_last, rfit_last = self.last_fit
                self.pub_lane.publish(Float32MultiArray(data=list(lfit_last) + list(rfit_last)))
            else:
                self.pub_lane.publish(Float32MultiArray(data=[]))
            

        if self.mstatus == "STOP":
            debug_frame = self._traffic_light(frame, debug_frame)
            
        if self.mstatus == "PARKING":
            res = self._ar_tag(frame, debug_frame)
            if res is not None:
                debug_frame = res

        combined_view = np.hstack([debug_frame, debug_bev])
        self._publish_overlay(combined_view)

    # ---------------- algorithms ----------------
    def _lane_stop(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        ymask = cv2.inRange(hsv, *self.YELLOW)
        wmask = cv2.inRange(hsv, *self.WHITE)

        h, w = ymask.shape
        stop_y0, stop_y1 = int(self.stop_y0 * h), int(self.stop_y1 * h)

        lane_mask = cv2.bitwise_or(ymask, wmask)

        gray_bev = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(gray_bev, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        _, stop_edge_mask = cv2.threshold(sobely, 35, 255, cv2.THRESH_BINARY)

        roi_stop_mask = lane_mask[stop_y0:stop_y1, :].copy()
        
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        lane_only_part = cv2.morphologyEx(roi_stop_mask, cv2.MORPH_OPEN, vert_kernel)
        
        pure_stop_candidate = cv2.subtract(roi_stop_mask, lane_only_part)
        pure_stop_candidate = cv2.bitwise_and(pure_stop_candidate, stop_edge_mask[stop_y0:stop_y1, :])

        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        final_stop_mask = cv2.morphologyEx(pure_stop_candidate, cv2.MORPH_OPEN, horiz_kernel)

        ypx = cv2.countNonZero(cv2.bitwise_and(final_stop_mask, ymask[stop_y0:stop_y1, :]))
        wpx = cv2.countNonZero(cv2.bitwise_and(final_stop_mask, wmask[stop_y0:stop_y1, :]))

        stop = "NONE"
        if ypx > w * self.stop_thr:
            stop = "YELLOW"
        elif wpx > w * self.stop_thr:
            stop = "WHITE"

        lane = lane_mask.copy()
        if stop != "NONE":
            lane[stop_y0:stop_y1, :][final_stop_mask > 0] = 0
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

        EXPECTED_L_X = int(self.w * (150/640))
        EXPECTED_R_X = int(self.w * (490/640))
        IMG_BOTTOM_Y = self.h - 1

        ANCHOR_WEIGHT = 30 

        l_fit_res, r_fit_res = None, None
        
        try:
            li = np.concatenate(lidx) if lidx else np.array([], np.int32)
            
            if len(li) >= self.minpts:
                real_y = nzy[li]
                real_x = nzx[li]

                anchor_y = np.full(ANCHOR_WEIGHT, IMG_BOTTOM_Y)
                anchor_x = np.full(ANCHOR_WEIGHT, EXPECTED_L_X)

                final_y = np.concatenate([real_y, anchor_y])
                final_x = np.concatenate([real_x, anchor_x])

                if len(li) < self.minpts * 1.5:
                    tmp_f = np.polyfit(final_y, final_x, 1)
                    l_fit_res = np.array([0.0, tmp_f[0], tmp_f[1]])
                else:
                    l_fit_res = np.polyfit(final_y, final_x, 2)
            
            ri = np.concatenate(ridx) if ridx else np.array([], np.int32)
            if len(ri) >= self.minpts:
                real_y = nzy[ri]
                real_x = nzx[ri]

                anchor_y = np.full(ANCHOR_WEIGHT, IMG_BOTTOM_Y)
                anchor_x = np.full(ANCHOR_WEIGHT, EXPECTED_R_X)

                final_y = np.concatenate([real_y, anchor_y])
                final_x = np.concatenate([real_x, anchor_x])

                r_fit_res = np.polyfit(final_y, final_x, 2)

        except Exception:
            pass

        return l_fit_res, r_fit_res

    @staticmethod
    def _poly_x(f, y):
        return (f[0] * y * y) + (f[1] * y) + f[2]

    def _debug_line(self, bev, lane, lf, rf, stop, ypx, wpx, color):
        debug_bev = bev.copy()
        debug_bev[:, :, 1] = np.maximum(debug_bev[:, :, 1], (lane // 2).astype(np.uint8))

        h, w = debug_bev.shape[:2]

        if color != "NONE":
            y0 = int(self.stop_y0 * h)
            y1 = int(self.stop_y1 * h)
            box_color = (0, 255, 255) if stop == "YELLOW" else (200, 200, 200)
            cv2.rectangle(debug_bev, (0, y0), (w, y1), box_color, 4)
            cv2.putText(debug_bev, f"{stop} LINE", (w // 2 - 100, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        ptsL, ptsR = [], []
        plot_y = np.linspace(h - 1, 0, h // 20).astype(int)

        if lf is not None:
            for y in plot_y:
                lx = self._poly_x(lf, y)
                if 0 <= lx < w:
                    ptsL.append((int(lx), y))
            if len(ptsL) >= 2:
                cv2.polylines(debug_bev, [np.array(ptsL)], False, (255, 0, 0), 2)

        if rf is not None:
            for y in plot_y:
                rx = self._poly_x(rf, y)
                if 0 <= rx < w:
                    ptsR.append((int(rx), y))
            if len(ptsR) >= 2:
                cv2.polylines(debug_bev, [np.array(ptsR)], False, (0, 0, 255), 2)

        # [수정] Center Path 시각화 (초록색 선)
        if len(self.center_path_pts) >= 2:
            cv2.polylines(debug_bev, [self.center_path_pts], False, (0, 255, 0), 2)
        pts = np.array([(479, 320), (360, 320)], dtype=np.int32)
        cv2.polylines(debug_bev, [pts], False, (255, 255, 0), 2)

        detect_status = []
        if lf is not None: detect_status.append("L")
        if rf is not None: detect_status.append("R")
        detect_str = "+".join(detect_status) if detect_status else "NONE"

        cv2.putText(debug_bev, f"{self.mdir} {self.mstatus} STOP:{stop} DET:{detect_str}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(debug_bev, f"Y:{ypx} W:{wpx}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        
        return debug_bev

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

        if light_result != "UNKNOWN":
            self._debug_light(debug_frame, 0, int(h * 0.4), int(w * 0.2), int(w * 0.8), light_result)
        return debug_frame

    def _ar_tag(self, frame, debug_frame):
        if not hasattr(cv2, "aruco"):
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ar = cv2.aruco
        dic = ar.getPredefinedDictionary(ar.DICT_6X6_250)
        prm = ar.DetectorParameters()
        detector = ar.ArucoDetector(dic, prm)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return

        cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids, (255, 0, 0))

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