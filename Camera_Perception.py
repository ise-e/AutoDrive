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
import time
from collections import deque, Counter

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, String


def now_sec() -> float:
    return time.time()


class _MajorityHysteresis:
    """Majority vote with simple hysteresis + UNKNOWN clearing.

    - Keeps last stable label unless a new label wins >= min_hits in the window.
    - UNKNOWN does not directly override; it clears after consecutive UNKNOWNs.
    """
    def __init__(self, window: int, min_hits: int, unknown_clear: int, default: str = "UNKNOWN"):
        self._buf = deque(maxlen=max(1, int(window)))
        self._min_hits = max(1, int(min_hits))
        self._unknown_clear = max(1, int(unknown_clear))
        self._default = str(default)
        self._stable = str(default)
        self._unknown_run = 0

    def update(self, raw: str) -> str:
        raw = str(raw)
        self._buf.append(raw)

        if raw == "UNKNOWN":
            self._unknown_run += 1
        else:
            self._unknown_run = 0

        counts = Counter([x for x in self._buf if x != "UNKNOWN"])
        if counts:
            cand, hits = counts.most_common(1)[0]
            if self._stable == self._default:
                if hits >= self._min_hits:
                    self._stable = cand
            else:
                if cand != self._stable and hits >= self._min_hits:
                    self._stable = cand

        if self._unknown_run >= self._unknown_clear:
            self._stable = self._default

        return self._stable


class _StoplineDebouncer:
    """Stopline debouncer with majority window + short hold time on hit."""
    def __init__(self, window: int, min_hits: int, hold_sec: float):
        self._buf = deque(maxlen=max(1, int(window)))
        self._min_hits = max(1, int(min_hits))
        self._hold_sec = max(0.0, float(hold_sec))
        self._last = "NONE"
        self._last_t = 0.0

    def update(self, raw: str, t: float) -> str:
        raw = str(raw)
        self._buf.append(raw)

        if raw != "NONE":
            counts = Counter([x for x in self._buf if x != "NONE"])
            if counts:
                cand, hits = counts.most_common(1)[0]
                if hits >= self._min_hits:
                    self._last = cand
                    self._last_t = float(t)
                    return self._last

        if self._last != "NONE" and (float(t) - float(self._last_t)) <= self._hold_sec:
            return self._last

        return "NONE"


class CameraPerception:
    def __init__(self):
        rospy.init_node("camera_perception_node")
        get_param = rospy.get_param

        # ---------- IO ----------
        self.cam_topic = get_param("~camera_topic", "/camera/image_raw")
        self.w, self.h = int(get_param("~out_w", 640)), int(get_param("~out_h", 480))

        self.overlay_topic = get_param("~overlay_topic", "/debug/lane_overlay/image/compressed")
        self.overlay_on = bool(get_param("~overlay_enable", False))
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
        self.stop_thr = float(get_param("~stop_px_per_col", 20.0))
        self.lane_cut = float(get_param("~lane_fit_ymax", 0.70))

        # 정지선 검증을 위한 파라미터
        self.stop_line_angle_thr = float(get_param("~stop_line_angle_threshold", 20.0))  # 수평선 허용 각도
        self.stop_line_min_length = float(get_param("~stop_line_min_length", 100.0))  # 최소 선 길이

        # ---------- stopline (row-projection + hough validation) ----------
        # Strong thresholds: 실제 정지선 확정에 사용
        self.stop_row_score_thr = float(get_param("~stop_row_score_thr", 0.30))
        self.stop_row_cov_thr = float(get_param("~stop_row_cov_thr", 0.38))

        # Hough 검증 파라미터: 수평선이 '진짜 선'인지 확인
        self.stop_hough_threshold = int(get_param("~stop_hough_threshold", 70))
        self.stop_hough_max_gap = int(get_param("~stop_hough_max_gap", 25))


        # ---------- dropout hold ----------
        self.hold_sec = float(get_param("~last_fit_hold_sec", 0.5))
        self.last_fit = None
        self.last_fit_t = 0.0


        # 단일 차선/갈림길 대비: 관측된 차선 폭(px) EMA
        self._lane_width_px = float(rospy.get_param('~lane_width_default_px', 360.0))
        self._lane_width_alpha = float(rospy.get_param('~lane_width_ema_alpha', 0.2))

        # 검지 실패 시, 마지막 차선을 잠깐 유지하는 홀드 시간(초)
        self._hold_lane_sec = float(rospy.get_param('~hold_lane_sec', 0.25))
        # ---------- mission state ----------
        self.mdir = "RIGHT"
        self.mstatus = "NONE"

        # ---------- HSV ranges ----------
        self.RED1 = (np.array([0, 100, 100]),   np.array([10, 255, 255]))
        self.RED2 = (np.array([160, 100, 100]), np.array([179, 255, 255]))
        self.YELLOW = (np.array([15, 100, 100]), np.array([35, 255, 255]))
        self.GREEN  = (np.array([45, 100, 100]), np.array([90, 255, 255]))
        self.WHITE  = (np.array([0, 0, 200]),    np.array([179, 40, 255]))
        # ---------- Lane color thresholds (tunable) ----------
        # You can override via ROS params:
        #  - ~yellow_lower_hsv / ~yellow_upper_hsv : [H,S,V]
        #  - ~white_v_min, ~white_s_max, ~white_v_percentile
        #  - ~lane_mask_kernel : odd int (e.g., 3 or 5)
        yl = get_param("~yellow_lower_hsv", None)
        yu = get_param("~yellow_upper_hsv", None)
        if isinstance(yl, (list, tuple)) and len(yl) == 3 and isinstance(yu, (list, tuple)) and len(yu) == 3:
            self.YELLOW = (np.array(yl, dtype=np.uint8), np.array(yu, dtype=np.uint8))
        
        self.white_v_min = int(get_param("~white_v_min", 170))
        self.white_s_max = int(get_param("~white_s_max", 80))
        self.white_v_percentile = float(get_param("~white_v_percentile", 75.0))
        
        self.use_lab_yellow = bool(get_param("~use_lab_yellow", True))
        self.lab_b_min = int(get_param("~lab_b_min", 145))
        self.lab_l_min = int(get_param("~lab_l_min", 80))
        
        self.lane_mask_kernel = int(get_param("~lane_mask_kernel", 3))
        if self.lane_mask_kernel < 1:
            self.lane_mask_kernel = 1
        if self.lane_mask_kernel % 2 == 0:
            self.lane_mask_kernel += 1

        # ---------- temporal filters ----------
        self._tl_filter = _MajorityHysteresis(
        window=rospy.get_param("~tl_window", 5),
        min_hits=rospy.get_param("~tl_min_hits", 3),
        unknown_clear=rospy.get_param("~tl_unknown_clear", 5),
        default="UNKNOWN",
        )
        self._stop_filter = _StoplineDebouncer(
        window=rospy.get_param("~stop_window", 5),
        min_hits=rospy.get_param("~stop_min_hits", 2),
        hold_sec=rospy.get_param("~stop_hold_sec", 0.30),
        )

        # ---------- traffic light ROI & coupling with stopline ----------
        # Base ROI (default): top tl_roi_y of the image, centered between [tl_roi_x0, tl_roi_x1]
        self.tl_roi_y = float(rospy.get_param("~tl_roi_y", 0.40))
        self.tl_roi_x0 = float(rospy.get_param("~tl_roi_x0", 0.20))
        self.tl_roi_x1 = float(rospy.get_param("~tl_roi_x1", 0.80))
        self.tl_pixel_thr = int(rospy.get_param("~tl_pixel_thr", 200))

        # Expanded ROI when we are near an intersection/stopline (helps early acquisition)
        self.tl_roi_y_expanded = float(rospy.get_param("~tl_roi_y_expanded", 0.60))
        self.tl_roi_x0_expanded = float(rospy.get_param("~tl_roi_x0_expanded", 0.10))
        self.tl_roi_x1_expanded = float(rospy.get_param("~tl_roi_x1_expanded", 0.90))
        self.tl_pixel_thr_expanded = int(rospy.get_param("~tl_pixel_thr_expanded", 160))

        # ---------- traffic light (blob scoring) ----------
        # ROI 내에서 '원형/밝은 덩어리'를 점수화해서 신호등을 더 안정적으로 인식합니다.
        self.tl_blob_score_thr_ratio = float(get_param("~tl_blob_score_thr_ratio", 0.0025))
        self.tl_blob_score_thr_ratio_expanded = float(get_param("~tl_blob_score_thr_ratio_expanded", 0.0012))
        self.tl_blob_min_area_ratio = float(get_param("~tl_blob_min_area_ratio", 0.00015))
        self.tl_blob_min_circularity = float(get_param("~tl_blob_min_circularity", 0.35))


        # If stopline is strongly detected OR weakly suspected, keep "near_intersection" True for this duration.
        self.near_intersection_hold_sec = float(rospy.get_param("~near_intersection_hold_sec", 1.5))

        # Weak stopline thresholds (used only to expand TL ROI; does NOT directly trigger a STOP decision).
        self.stop_row_score_weak_thr = float(rospy.get_param("~stop_row_score_weak_thr", 0.10))
        self.stop_row_cov_weak_thr = float(rospy.get_param("~stop_row_cov_weak_thr", 0.20))

        self._near_intersection_until = 0.0

        # ---------- fork (갈림길) 판단 파라미터 ----------
        # pdf 규칙:
        #  - WHITE 정지선 전: (노란 차선 1개 + 흰 차선 1개)로 갈림
        #  - YELLOW 정지선 전: (노란 차선 2개)로 갈림
        self.fork_check_y0 = float(rospy.get_param("~fork_check_y0", 0.40))
        self.fork_check_y1 = float(rospy.get_param("~fork_check_y1", 0.68))
        self.fork_smooth_win = int(max(5, int(rospy.get_param("~fork_smooth_win", 21)) | 1))  # odd
        self.fork_peak_min_dist_px = int(max(20, int(rospy.get_param("~fork_peak_min_dist_px", 70))))
        self.fork_peak_rel_thr = float(rospy.get_param("~fork_peak_rel_thr", 0.35))
        self.fork_col_thr_per_row = float(rospy.get_param("~fork_col_thr_per_row", 0.06))
        self.fork_peak_sep_thr = float(rospy.get_param("~fork_peak_sep_thr", 120.0))
        self.fork_width_dev_thr = float(rospy.get_param("~fork_width_deviation_thr", 60.0))
        self.fork_top_min_width = float(rospy.get_param("~fork_top_min_width", 60.0))
        self.fork_y_ref_ratio = float(rospy.get_param("~fork_y_ref_ratio", 0.55))
        self.fork_use_color_guidance = bool(rospy.get_param("~fork_use_color_guidance", True))



        # ---------- pubs ----------
        self.pub_lane = rospy.Publisher("/lane_coeffs", Float32MultiArray, queue_size=1)
        self.pub_stop = rospy.Publisher("/stop_line_status", String, queue_size=1)
        self.pub_tl   = rospy.Publisher("/traffic_light_status", String, queue_size=1)
        self.pub_ar   = rospy.Publisher("/ar_tag_info", Float32MultiArray, queue_size=1)
        self.pub_ov   = rospy.Publisher(self.overlay_topic, CompressedImage, queue_size=1)

        # ---------- subs ----------
        # main.sh / launch에 따라 카메라 토픽이 달라질 수 있어 후보들을 모두 구독합니다.
        # 첫 프레임이 들어온 토픽을 active로 고정하고 나머지는 무시합니다.
        self._active_cam_topic = None
        cam_topics = rospy.get_param("~camera_topics", [
            self.cam_topic,
            "camera/image_raw",
            "/usb_cam/image_raw",
            "/orda/usb_cam/image_raw",
            "/camera/color/image_raw",
            "/image_raw",
        ])
        self._cam_subs = []
        for t in cam_topics:
            if not t:
                continue
            try:
                self._cam_subs.append(rospy.Subscriber(t, Image, lambda m, _t=t: self._on_img(m, _t), queue_size=1))
            except Exception as e:
                rospy.logwarn(f"[CameraPerception] Failed to subscribe {t}: {e}")
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
    def _update_lane_width(self, lfit, rfit) -> None:
        """Update EMA of lane width in pixels using current left/right fits.

        The fits are x(y) polynomials in BEV image coordinates.
        """
        try:
            y_bot = self.h - 1
            y_mid = int(self.h * 0.6)
            y_top = int(self.h * 0.3)
            widths = []
            for y in (y_bot, y_mid, y_top):
                lx = float(self._poly_x(lfit, y))
                rx = float(self._poly_x(rfit, y))
                w = rx - lx
                if w > 0:
                    widths.append(w)
            if not widths:
                return
            w_new = float(np.median(widths))
            # sanity: ignore nonsense measurements
            if w_new < 60 or w_new > self.w * 1.5:
                return
            a = float(self._lane_width_alpha)
            self._lane_width_px = (1.0 - a) * float(self._lane_width_px) + a * w_new
        except Exception:
            # keep running even if width update fails
            return

    def _synthesize_pair_from_single(self, fit_single):
        """Given a single lane polynomial, synthesize the missing side using lane width EMA."""
        if fit_single is None:
            return None, None
        a, b, c = float(fit_single[0]), float(fit_single[1]), float(fit_single[2])
        y_ref = self.h - 1
        x_ref = float(self._poly_x((a, b, c), y_ref))
        delta = float(self._lane_width_px)
        # decide side by x position at bottom
        if x_ref < self.w * 0.5:
            # treat as left lane => synth right by shifting +delta
            lfit = np.array([a, b, c], dtype=np.float32)
            rfit = np.array([a, b, c + delta], dtype=np.float32)
        else:
            # treat as right lane => synth left by shifting -delta
            lfit = np.array([a, b, c - delta], dtype=np.float32)
            rfit = np.array([a, b, c], dtype=np.float32)
        return lfit, rfit

    def _on_img(self, msg: Image, topic: str = ""):
        if self._active_cam_topic is None:
            self._active_cam_topic = topic or self.cam_topic
            rospy.loginfo(f"[CameraPerception] Active camera topic: {self._active_cam_topic}")
        elif topic and topic != self._active_cam_topic:
            return

        frame = self._rosimg_to_bgr(msg)
        if frame is None:
            return

        frame = cv2.resize(frame, (self.w, self.h))
        debug_frame = frame.copy()
        bev = cv2.warpPerspective(frame, self.M, (self.w, self.h))

        lfit, rfit, stop, debug_bev = self._lane_stop(bev)
        stop_f = self._stop_filter.update(stop, now_sec())
        self.pub_stop.publish(stop_f)

        # 양쪽 차선 모두 검지
        if lfit is not None and rfit is not None:
            self._update_lane_width(lfit, rfit)
            # 이미지 최하단
            y_bot = self.h - 1
            # 이미지 상단
            y_top = int(self.h * 0.3)
            # 각 지점에서의 너비 계산
            w_bot = self._poly_x(rfit, y_bot) - self._poly_x(lfit, y_bot)
            w_top = self._poly_x(rfit, y_top) - self._poly_x(lfit, y_top)
            # 너비 변화량 (Width Deviation)
            width_deviation = abs(w_top - w_bot)

            # --- fork pattern (pdf 규칙 기반) ---
            ymask, wmask, _lane_mask_dbg = self._make_lane_masks(bev)
            fork_info = self._fork_pattern_from_masks(ymask, wmask, stop_f)

            need_fork = bool(fork_info.get("detected")) and (
                (width_deviation > float(self.fork_width_dev_thr)) or
                (w_top < float(self.fork_top_min_width)) or
                (float(fork_info.get("sep", 0.0)) > float(self.fork_peak_sep_thr))
            )

            if need_fork:
                if self.fork_use_color_guidance and (w_top >= w_bot):
                    target = self._select_fork_target_peak(fork_info)
                    if target is not None:
                        _c, x_target = target
                        y_ref = int(self.h * float(self.fork_y_ref_ratio))
                        base_fit = self._choose_fit_near_x(lfit, rfit, float(x_target), y_ref)
                        l_syn, r_syn = self._synthesize_pair_from_single(base_fit)
                        self.pub_lane.publish(Float32MultiArray(data=list(l_syn) + list(r_syn)))
                    else:
                        # 피크가 애매하면 기존 방식 fallback
                        base = lfit if self.mdir == "LEFT" else rfit
                        l_syn, r_syn = self._synthesize_pair_from_single(base)
                        self.pub_lane.publish(Float32MultiArray(data=list(l_syn) + list(r_syn)))
                else:
                    # 수렴/예외: 기존 폭 기반 처리 유지
                    if w_top < w_bot:
                        base = rfit if self.mdir == "LEFT" else lfit
                    else:
                        base = lfit if self.mdir == "LEFT" else rfit
                    l_syn, r_syn = self._synthesize_pair_from_single(base)
                    self.pub_lane.publish(Float32MultiArray(data=list(l_syn) + list(r_syn)))
            else:
                self.last_fit, self.last_fit_t = (lfit, rfit), now_sec()
                self.pub_lane.publish(Float32MultiArray(data=list(lfit) + list(rfit)))

        # 왼쪽만 검지
        elif lfit is not None:
            l_syn, r_syn = self._synthesize_pair_from_single(lfit)
            self.pub_lane.publish(Float32MultiArray(data=list(l_syn) + list(r_syn)))
        # 오른쪽만 검지
        elif rfit is not None:
            l_syn, r_syn = self._synthesize_pair_from_single(rfit)
            self.pub_lane.publish(Float32MultiArray(data=list(l_syn) + list(r_syn)))
        # 둘 다 미검지 ** 이 부분은 나중에 판단 노드로 옮길 생각입니다
        else:
            if self.last_fit is not None and (now_sec() - self.last_fit_t) < self.hold_sec:
                lfit_last, rfit_last = self.last_fit
                self.pub_lane.publish(Float32MultiArray(data=list(lfit_last) + list(rfit_last)))
            else:
                # 검지 실패 시: 마지막 차선을 잠깐 유지 (주행 흔들림 감소)
                if self.last_fit is not None and (now_sec() - float(self.last_fit_t)) <= float(self._hold_lane_sec):
                    lfit, rfit = self.last_fit
                    self.pub_lane.publish(Float32MultiArray(data=list(lfit) + list(rfit)))
                else:
                    self.pub_lane.publish(Float32MultiArray(data=[]))
        expanded_tl = self._near_intersection_active()
        if self.mstatus == "STOP" or expanded_tl:
            debug_frame = self._traffic_light(frame, debug_frame, expanded=expanded_tl)
            
        if self.mstatus == "PARKING":
            res = self._ar_tag(frame, debug_frame)
            if res is not None:
                debug_frame = res
        if debug_bev is None:
            debug_bev = bev.copy()
        combined_view = np.hstack([debug_frame, debug_bev])
        self._publish_overlay(combined_view)

    # ---------------- algorithms (원본 로직 유지) ----------------
    def _make_lane_masks(self, bev_bgr):
        """Return (ymask, wmask, lane_mask) in BEV coordinates.

        - Yellow: HSV inRange (+ optional LAB b-channel support)
        - White : adaptive V threshold + low S threshold
        """
        h, w = bev_bgr.shape[:2]
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # --- Yellow mask ---
        ymask_hsv = cv2.inRange(hsv, *self.YELLOW)

        if self.use_lab_yellow:
            lab = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            ymask_lab = cv2.inRange(B, self.lab_b_min, 255) & cv2.inRange(L, self.lab_l_min, 255)
            ymask = cv2.bitwise_or(ymask_hsv, ymask_lab)
        else:
            ymask = ymask_hsv

        # --- White mask (adaptive) ---
        y0 = int(h * 0.45)
        roi_v = V[y0:, :]
        v_thr = int(max(self.white_v_min, np.percentile(roi_v, self.white_v_percentile)))
        wmask = cv2.inRange(V, v_thr, 255) & cv2.inRange(S, 0, self.white_s_max)

        lane_mask = cv2.bitwise_or(ymask, wmask)

        # Morphology: connect broken paint, suppress speckles
        k = self.lane_mask_kernel
        if k >= 3:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, ker, iterations=1)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, ker, iterations=1)

        return ymask, wmask, lane_mask


    def _smooth_1d(self, arr, win: int):
        if win <= 1:
            return arr
        k = np.ones((win,), dtype=np.float32) / float(win)
        return np.convolve(arr.astype(np.float32), k, mode="same")

    def _find_peaks_1d(self, hist, rel_thr: float, min_dist: int):
        """local maxima + greedy distance suppression"""
        if hist is None or len(hist) == 0:
            return []
        h = np.asarray(hist, dtype=np.float32)
        mx = float(h.max()) if h.size else 0.0
        if mx <= 1e-6:
            return []
        thr = mx * float(rel_thr)
        cand = []
        for i in range(1, len(h) - 1):
            if h[i] >= thr and h[i] >= h[i - 1] and h[i] >= h[i + 1]:
                cand.append((float(h[i]), int(i)))
        cand.sort(reverse=True)  # by height
        picked = []
        for _, x in cand:
            if all(abs(x - px) >= int(min_dist) for px in picked):
                picked.append(x)
        picked.sort()
        return picked

    def _fork_pattern_from_masks(self, ymask, wmask, stop_color: str):
        """정지선 색 규칙 기반 갈림길 패턴 감지"""
        if stop_color not in ("WHITE", "YELLOW"):
            return {"detected": False, "peaks_y": [], "peaks_w": [], "peaks_all": [], "sep": 0.0}

        h, _w = ymask.shape[:2]
        y0 = int(h * float(self.fork_check_y0))
        y1 = int(h * float(self.fork_check_y1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        y_roi = ymask[y0:y1, :]
        w_roi = wmask[y0:y1, :]

        # column hist: ROI 내에서 해당 column이 얼마나 '지속적으로' 켜져있는지
        y_hist = np.count_nonzero(y_roi > 0, axis=0).astype(np.float32)
        w_hist = np.count_nonzero(w_roi > 0, axis=0).astype(np.float32)

        roi_h = float(max(1, (y1 - y0)))
        y_hist = y_hist / roi_h
        w_hist = w_hist / roi_h

        col_thr = float(self.fork_col_thr_per_row)
        y_hist[y_hist < col_thr] = 0.0
        w_hist[w_hist < col_thr] = 0.0

        y_hist = self._smooth_1d(y_hist, int(self.fork_smooth_win))
        w_hist = self._smooth_1d(w_hist, int(self.fork_smooth_win))

        peaks_y = self._find_peaks_1d(y_hist, float(self.fork_peak_rel_thr), int(self.fork_peak_min_dist_px))
        peaks_w = self._find_peaks_1d(w_hist, float(self.fork_peak_rel_thr), int(self.fork_peak_min_dist_px))

        candidates = []
        if stop_color == "WHITE":
            candidates += [("YELLOW", x) for x in peaks_y]
            candidates += [("WHITE", x) for x in peaks_w]
            detected = (len(peaks_y) >= 1 and len(peaks_w) >= 1 and len(candidates) >= 2)
        else:
            candidates += [("YELLOW", x) for x in peaks_y]
            detected = (len(peaks_y) >= 2)

        candidates.sort(key=lambda t: t[1])
        sep = float(candidates[-1][1] - candidates[0][1]) if len(candidates) >= 2 else 0.0

        return {"detected": bool(detected), "peaks_y": peaks_y, "peaks_w": peaks_w, "peaks_all": candidates, "sep": sep}

    def _select_fork_target_peak(self, fork_info: dict):
        peaks_all = list(fork_info.get("peaks_all") or [])
        if len(peaks_all) < 2:
            return None
        peaks_all.sort(key=lambda t: t[1])
        return peaks_all[0] if self.mdir == "LEFT" else peaks_all[-1]

    def _choose_fit_near_x(self, lfit, rfit, x_target: float, y_ref: int):
        xl = float(self._poly_x(lfit, y_ref))
        xr = float(self._poly_x(rfit, y_ref))
        return lfit if abs(x_target - xl) <= abs(x_target - xr) else rfit


    def _hough_has_horizontal(self, edge_band: np.ndarray) -> bool:
        """Return True if a sufficiently-long near-horizontal segment exists in edge_band."""
        lines = cv2.HoughLinesP(
            edge_band,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(self.stop_hough_threshold),
            minLineLength=int(self.stop_line_min_length),
            maxLineGap=int(self.stop_hough_max_gap),
        )
        if lines is None:
            return False

        ang_thr = float(self.stop_line_angle_thr)  # degrees
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < float(self.stop_line_min_length):
                continue
            ang = abs(float(np.degrees(np.arctan2(dy, dx))))
            ang = ang if ang <= 90.0 else 180.0 - ang
            if ang <= ang_thr:
                return True
        return False

    def _detect_stopline_rowproj(self, bev_bgr, ymask, wmask, roi_y0, roi_y1):
        """Robust stopline detection using BEV + row projection of horizontal edge energy."""
        h, w = bev_bgr.shape[:2]
        roi_y0 = int(max(0, min(h - 1, roi_y0)))
        roi_y1 = int(max(roi_y0 + 1, min(h, roi_y1)))

        roi = bev_bgr[roi_y0:roi_y1, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        sob = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_sob = cv2.convertScaleAbs(sob)

        thr = max(25, int(np.mean(abs_sob) + 1.0 * np.std(abs_sob)))
        edge = cv2.inRange(abs_sob, thr, 255)

        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 20), 3))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, hk, iterations=1)

        row_score = edge.sum(axis=1).astype(np.float32) / 255.0
        if len(row_score) >= 5:
            kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
            kernel /= kernel.sum()
            row_score = np.convolve(row_score, kernel, mode="same")

        peak_y = int(np.argmax(row_score))
        peak = float(row_score[peak_y])
        peak_norm = peak / float(w)

        band = 4
        y_a = max(0, peak_y - band)
        y_b = min(edge.shape[0], peak_y + band + 1)
        band_edge = edge[y_a:y_b, :]
        cov = float(np.count_nonzero(np.any(band_edge > 0, axis=0))) / float(w)

        score_thr = float(self.stop_row_score_thr)
        cov_thr = float(self.stop_row_cov_thr)

        hough_ok = False

        stop = "NONE"
        stop_mask = None

        if peak_norm >= score_thr and cov >= cov_thr:
            hough_ok = self._hough_has_horizontal(band_edge)
            if not hough_ok:
                dbg = {"peak_norm": peak_norm, "cov": cov, "thr": thr, "hough": False}
                return "NONE", None, dbg
            yy0 = roi_y0 + y_a
            yy1 = roi_y0 + y_b
            ycnt = int(cv2.countNonZero(ymask[yy0:yy1, :]))
            wcnt = int(cv2.countNonZero(wmask[yy0:yy1, :]))
            stop = "YELLOW" if ycnt > wcnt else "WHITE"

            stop_mask = np.zeros((h, w), dtype=np.uint8)
            stop_mask[yy0:yy1, :] = 255

        dbg = {"peak_norm": peak_norm, "cov": cov, "thr": thr, "hough": bool(hough_ok)}
        return stop, stop_mask, dbg

    def _lane_stop(self, bev):
        h, w = bev.shape[:2]

        # 1) Lane masks (yellow/white) with robustness improvements
        ymask, wmask, lane_mask = self._make_lane_masks(bev)

        # 2) Stopline detection ROI
        y0 = int(self.stop_y0 * h)
        y1 = int(self.stop_y1 * h)
        if y1 <= y0:
            y1 = min(h, y0 + 1)

        stop, stop_mask, dbg_stop = self._detect_stopline_rowproj(bev, ymask, wmask, y0, y1)

        # 2.5) Couple stopline <-> traffic light:
        # If we see (or even weakly suspect) a stopline, temporarily expand TL ROI to avoid missing the light.
        now_sec = rospy.Time.now().to_sec()
        weak_near = (
            float(dbg_stop.get("peak_norm", 0.0)) >= self.stop_row_score_weak_thr
            and float(dbg_stop.get("cov", 0.0)) >= self.stop_row_cov_weak_thr
        )
        if stop != "NONE" or weak_near:
            self._near_intersection_until = max(self._near_intersection_until, now_sec + self.near_intersection_hold_sec)

        # 3) Remove stopline region from lane mask (prevents polyfit corruption)
        clean_lane = lane_mask.copy()
        if stop_mask is not None:
            clean_lane[stop_mask > 0] = 0

        # 4) Crop top region for fitting stability (keep only nearer region)
        cut = int(self.lane_cut * h)
        clean_lane[cut:, :] = 0

        # 5) Fit lanes
        lfit, rfit = self._fit(clean_lane)

        # debug_bev는 hstack 때문에 무조건 ndarray로 만들어야 안전함
        try:
            ypx = int(dbg_stop.get("peak_norm", 0.0) * 100.0)
            wpx = int(dbg_stop.get("cov", 0.0) * 100.0)
        except Exception:
            ypx, wpx = 0, 0

        color = stop if stop != "NONE" else "NONE"
        debug_bev = self._debug_line(bev, clean_lane, lfit, rfit, stop, ypx, wpx, color)


        # (optional) lightweight debug overlay
        if debug_bev is not None:
            cv2.rectangle(debug_bev, (0, y0), (w - 1, y1 - 1), (50, 200, 50), 1)
            cv2.putText(
                debug_bev,
                f"STOP:{stop} score={dbg_stop['peak_norm']:.2f} cov={dbg_stop['cov']:.2f}",
                (10, max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

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

        EXPECTED_L_X = int(self.w * (150/640))  # 약 150px
        EXPECTED_R_X = int(self.w * (490/640))  # 약 490px
        IMG_BOTTOM_Y = self.h - 1               # 479px

        # 가중치 (앵커 점을 몇 개나 추가할지) - 높을수록 고정력 강함
        # 차가 흔들릴 때 유연성을 주려면 10~20, 강력하게 고정하려면 50 이상
        ANCHOR_WEIGHT = 30 

        l_fit_res, r_fit_res = None, None

        try:
            li = np.concatenate(lidx) if lidx else np.array([], np.int32)

            # [왼쪽 차선 피팅]
            if len(li) >= self.minpts:
                # 1. 실제 검출된 점들
                real_y = nzy[li]
                real_x = nzx[li]

                # 2. 앵커(고정) 점 생성: 맨 아래(IMG_BOTTOM_Y)에 EXPECTED_L_X 좌표를 여러 개 추가
                anchor_y = np.full(ANCHOR_WEIGHT, IMG_BOTTOM_Y)
                anchor_x = np.full(ANCHOR_WEIGHT, EXPECTED_L_X)

                # 3. 데이터 합치기 (실제 점 + 앵커 점)
                final_y = np.concatenate([real_y, anchor_y])
                final_x = np.concatenate([real_x, anchor_x])

                # 4. 합친 데이터로 피팅
                if len(li) < self.minpts * 1.5:
                    tmp_f = np.polyfit(final_y, final_x, 1)
                    l_fit_res = np.array([0.0, tmp_f[0], tmp_f[1]])
                else:
                    l_fit_res = np.polyfit(final_y, final_x, 2)

            # [오른쪽 차선 피팅]
            ri = np.concatenate(ridx) if ridx else np.array([], np.int32)
            if len(ri) >= self.minpts:
                real_y = nzy[ri]
                real_x = nzx[ri]

                # 앵커 추가
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
        # 차선 마스크(흰/노랑)를 초록 채널에 섞어서 표시
        debug_bev[:, :, 1] = np.maximum(debug_bev[:, :, 1], (lane // 2).astype(np.uint8))

        h, w = debug_bev.shape[:2]

        # 1. 정지선 박스 그리기 
        if color != "NONE":
            y0 = int(self.stop_y0 * h)
            y1 = int(self.stop_y1 * h)
            box_color = (0, 255, 255) if stop == "YELLOW" else (200, 200, 200)
            cv2.rectangle(debug_bev, (0, y0), (w, y1), box_color, 4)
            cv2.putText(debug_bev, f"{stop} LINE", (w // 2 - 100, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        # 2. 차선 시각화 (각각 독립적으로 수행)
        ptsL, ptsR = [], []

        # Y좌표 생성 (이미지 하단 -> 상단)
        plot_y = np.linspace(h - 1, 0, h // 20).astype(int)

        LINE_WIDTH_PX = 400
        if lf is None or rf is None:
            base_lane = rf if lf is None else lf
            if base_lane is None: base_lane = [0, 0, w/2+200]
            grad = -base_lane[0]*h**2+base_lane[1]*h
            if grad < 0:
                if rf is None:
                    rf = base_lane
                    lf = (base_lane[0], base_lane[1], base_lane[2]-LINE_WIDTH_PX)
                else:
                    lf = base_lane
                    rf = (base_lane[0], base_lane[1], base_lane[2]+LINE_WIDTH_PX)
            else:
                if lf is None:
                    lf = base_lane
                    rf = (base_lane[0], base_lane[1], base_lane[2]+LINE_WIDTH_PX)
                else:
                    rf = base_lane
                    lf = (base_lane[0], base_lane[1], base_lane[2]-LINE_WIDTH_PX)

        # (1) 왼쪽 차선 그리기
        if lf is not None:
            for y in plot_y:
                lx = self._poly_x(lf, y)
                if 0 <= lx < w:
                    ptsL.append((int(lx), y))
            if len(ptsL) >= 2:
                cv2.polylines(debug_bev, [np.array(ptsL)], False, (255, 0, 0), 2)  # 파란색

        # (2) 오른쪽 차선 그리기
        if rf is not None:
            for y in plot_y:
                rx = self._poly_x(rf, y)
                if 0 <= rx < w:
                    ptsR.append((int(rx), y))
            if len(ptsR) >= 2:
                cv2.polylines(debug_bev, [np.array(ptsR)], False, (0, 0, 255), 2)  # 빨간색

        # (3) 주행 중심선 그리기 (양쪽 다 있을 때만)
        ptsC = []
        for y in plot_y:
            lx = self._poly_x(lf, y)
            rx = self._poly_x(rf, y)
            cx = (lx + rx) * 0.5
            if 0 <= cx < w:
                ptsC.append((int(cx), y))

        if len(ptsC) >= 2:
            cv2.polylines(debug_bev, [np.array(ptsC)], False, (0, 255, 0), 2)  # 초록색
        cv2.polylines(debug_bev, [np.array([(w//2, h), (w//2, h*2//3)])], False, (0, 255, 255), 2)
        # 3. 상태 텍스트
        detect_status = []
        if lf is not None: detect_status.append("L")
        if rf is not None: detect_status.append("R")
        detect_str = "+".join(detect_status) if detect_status else "NONE"

        cv2.putText(debug_bev, f"{self.mdir} {self.mstatus} STOP:{stop} DET:{detect_str}",
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

    def _near_intersection_active(self) -> bool:
        try:
            return rospy.Time.now().to_sec() <= float(getattr(self, "_near_intersection_until", 0.0))
        except Exception:
            return False

    def _get_tl_roi(self, hsv, expanded: bool):
        h, w = hsv.shape[:2]

        if expanded:
            y_ratio = self.tl_roi_y_expanded
            x0_ratio = self.tl_roi_x0_expanded
            x1_ratio = self.tl_roi_x1_expanded
        else:
            y_ratio = self.tl_roi_y
            x0_ratio = self.tl_roi_x0
            x1_ratio = self.tl_roi_x1

        # Clamp ratios
        y_ratio = max(0.05, min(1.0, float(y_ratio)))
        x0_ratio = max(0.0, min(1.0, float(x0_ratio)))
        x1_ratio = max(0.0, min(1.0, float(x1_ratio)))
        if x1_ratio <= x0_ratio:
            x0_ratio, x1_ratio = 0.20, 0.80  # fallback

        y1 = int(h * y_ratio)
        x0 = int(w * x0_ratio)
        x1 = int(w * x1_ratio)
        y1 = max(1, min(h, y1))
        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))

        roi = hsv[0:y1, x0:x1]
        return roi, (0, y1, x0, x1)

    def _best_blob_score(self, mask: np.ndarray) -> float:
        """Blob scoring: returns best score among candidate blobs in the binary mask."""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0

        H, W = m.shape[:2]
        roi_area = float(H * W)
        min_area = max(12.0, roi_area * float(self.tl_blob_min_area_ratio))
        min_circ = float(self.tl_blob_min_circularity)

        best = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / float(h + 1e-6)
            if ar < 0.4 or ar > 2.5:
                continue
            peri = float(cv2.arcLength(c, True))
            if peri <= 1e-6:
                continue
            circ = 4.0 * np.pi * area / (peri * peri)
            if circ < min_circ:
                continue

            score = area * (0.5 + circ)
            if score > best:
                best = score
        return float(best)

    def _traffic_light(self, frame, debug_frame, expanded: bool = False):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi, (y0, y1, x0, x1) = self._get_tl_roi(hsv, expanded)

        m_red = cv2.bitwise_or(cv2.inRange(roi, *self.RED1), cv2.inRange(roi, *self.RED2))
        m_yel = cv2.inRange(roi, *self.YELLOW)
        m_grn = cv2.inRange(roi, *self.GREEN)

        s_red = self._best_blob_score(m_red)
        s_yel = self._best_blob_score(m_yel)
        s_grn = self._best_blob_score(m_grn)

        best_score, best_label = max((s_red, "RED"), (s_yel, "YELLOW"), (s_grn, "GREEN"))

        # ROI 면적 기준 임계값(확장 ROI에서는 더 낮은 ratio 사용)
        H, W = roi.shape[:2]
        roi_area = float(H * W)
        ratio = float(self.tl_blob_score_thr_ratio_expanded if expanded else self.tl_blob_score_thr_ratio)
        default_thr = roi_area * ratio
        score_thr = float(rospy.get_param("~tl_blob_score_thr", default_thr))

        raw = best_label if best_score >= score_thr else "UNKNOWN"
        stable = self._tl_filter.update(raw)
        self.pub_tl.publish(stable)

        if stable != "UNKNOWN":
            self._debug_light(debug_frame, y0, y1, x0, x1, stable)
        return debug_frame

    def _ar_tag(self, frame, debug_frame):
        if not hasattr(cv2, "aruco"):
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ar = cv2.aruco
        dic = ar.getPredefinedDictionary(ar.DICT_5X5_1000)
        prm = ar.DetectorParameters()
        detector = ar.ArucoDetector(dic, prm)
        corners, ids, _ = detector.detectMarkers(gray)
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
