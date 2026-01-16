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
        self.stop_y0 = float(get_param("~stop_check_y0", 0.60))
        self.stop_y1 = float(get_param("~stop_check_y1", 0.85))
        self.stop_thr = float(get_param("~stop_px_per_col", 0.2))
        self.lane_cut = float(get_param("~lane_fit_ymax", 0.70))

        # 정지선 검증을 위한 파라미터
        self.stop_line_angle_thr = float(get_param("~stop_line_angle_threshold", 20.0))  # 수평선 허용 각도
        self.stop_line_min_length = float(get_param("~stop_line_min_length", 100.0))  # 최소 선 길이

        # ---------- dropout hold ----------
        self.hold_sec = float(get_param("~last_fit_hold_sec", 0.5))
        self.last_fit = None
        self.last_fit_t = 0.0
        
        # ---------- mission state ----------
        self.mdir = "RIGHT"
        self.mstatus = "NONE"
        self.is_white = False

        # ---------- HSV ranges ----------
        self.RED1 = (np.array([0, 100, 100]),   np.array([10, 255, 255]))
        self.RED2 = (np.array([160, 100, 100]), np.array([179, 255, 255]))
        self.YELLOW = (np.array([10, 40, 50]), np.array([45, 255, 255]))
        self.GREEN  = (np.array([45, 100, 100]), np.array([90, 255, 255]))
        self.WHITE  = (np.array([0, 0, 200]),    np.array([179, 40, 255]))

        # ---------- pubs ----------
        self.pub_lane = rospy.Publisher("/lane_coeffs", Float32MultiArray, queue_size=1)
        self.pub_stop = rospy.Publisher("/stop_line_status", String, queue_size=1)
        self.pub_tl   = rospy.Publisher("/traffic_light_status", String, queue_size=1)
        self.pub_ar   = rospy.Publisher("/ar_tag_info", Float32MultiArray, queue_size=1)
        self.pub_ov   = rospy.Publisher(self.overlay_topic, CompressedImage, queue_size=1)

        self.debug_timer = 0.0
        self.debug_interval = 0.2
        # ---------- subs ----------
        rospy.Subscriber(self.cam_topic, Image, self._on_img, queue_size=1, buff_size=2**24)
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
        
        now = now_sec()
        if now - self.debug_timer < self.debug_interval:
            return
        
        self.debug_timer = now

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

        def safe_list(fit):
            return list(fit) if fit is not None else []

        # 양쪽 차선 모두 검지
        if lfit is not None and rfit is not None:
            # 이미지 최하단
            y_bot = self.h - 1
            # 이미지 상단
            y_top = int(self.h * 0.3)
            # 이미지 제어 지점
            y_eval = self.h * 0.8
            # 각 지점에서의 너비 계산
            w_bot = self._poly_x(rfit, y_bot) - self._poly_x(lfit, y_bot)
            w_top = self._poly_x(rfit, y_top) - self._poly_x(lfit, y_top)
            # 너비 변화량 (Width Deviation)
            width_deviation = abs(w_top - w_bot)
            # 차이가 60 이상일 경우 **값 조정 필요
            if width_deviation > 50 or w_top < 40 or w_bot < 40:
                get_deriv = lambda f, y: 2 * f[0] * y + f[1]
                
                deriv_l = get_deriv(lfit, y_eval)
                deriv_r = get_deriv(rfit, y_eval)

                if self.mdir == "LEFT":
                    if (deriv_l - deriv_r) > 0 :
                        self.last_fit, self.last_fit_t = (lfit, None), now_sec()
                        self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
                    else :
                        self.last_fit, self.last_fit_t = (None, rfit), now_sec()
                        self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
                else: # RIGHT 미션
                    if (deriv_l - deriv_r) < 0 :
                        self.last_fit, self.last_fit_t = (lfit, None), now_sec()
                        self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
                    else :
                        self.last_fit, self.last_fit_t = (None, rfit), now_sec()
                        self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
            # 정상적인 경우
            else :
                self.last_fit, self.last_fit_t = (lfit, rfit), now_sec()
                self.pub_lane.publish(Float32MultiArray(data=list(lfit) + list(rfit)))
        # 왼쪽만 검지
        elif lfit is not None:
            self.last_fit, self.last_fit_t = (lfit, None), now_sec()
            self.pub_lane.publish(Float32MultiArray(data=list(lfit)))
        # 오른쪽만 검지
        elif rfit is not None:
            self.last_fit, self.last_fit_t = (None, rfit), now_sec()
            self.pub_lane.publish(Float32MultiArray(data=list(rfit)))
        # 둘 다 미검지 ** 이 부분은 나중에 판단 노드로 옮길 생각입니다
        else:
            if self.last_fit is not None and (now_sec() - self.last_fit_t) < self.hold_sec:
                l_last, r_last = self.last_fit
                # safe_list를 사용하여 None 에러 방지
                self.pub_lane.publish(Float32MultiArray(data=safe_list(l_last) + safe_list(r_last)))
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

    # ---------------- algorithms (원본 로직 유지) ----------------
    def _lane_stop(self, bev):
        h, w = bev.shape[:2]
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        # 1. 마스크 생성 (한 번만)
        ymask = cv2.inRange(hsv, *self.YELLOW)
        wmask = cv2.inRange(hsv, *self.WHITE)
        # 정지선용 통합 마스크
        lane_mask = cv2.bitwise_or(ymask, wmask)
        
        # 2. 정지선 ROI 설정
        y0 = int(self.stop_y0 * h)
        y1 = int(self.stop_y1 * h)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        
        # 3. ROI 영역에서 정지선 탐지
        roi_lane = lane_mask[y0:y1, :]
        
        # 수평 커널로 세로 차선 제거 (기울어진 정지선도 고려)
        # 20도 기울기 대응: 약간 더 관대한 커널 사용
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
        stop_roi = cv2.morphologyEx(roi_lane, cv2.MORPH_OPEN, h_kernel)
        stop_roi = cv2.morphologyEx(stop_roi, cv2.MORPH_CLOSE, h_kernel)

        # 4. 정지선 컨투어 추출
        contours, _ = cv2.findContours(stop_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stop = "NONE"
        stop_mask = None
        area_thr = w * (y1 - y0) * self.stop_thr
        width_thr = w * 0.3

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= area_thr:
                continue
                
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # 조건 체크 (기울어진 흰색 정지선 대응)
            if bh == 0 or bw <= width_thr:
                continue
                
            aspect_ratio = bw / float(bh)
            # 20도 기울기: aspect_ratio ≈ 2.75, 여유값 2.0으로 완화
            if aspect_ratio <= 3.0:
                continue
            
            # 색상 판별 (ROI 슬라이싱 한 번만)
            roi_y_slice = slice(y0 + y, y0 + y + bh)
            roi_x_slice = slice(x, x + bw)
            
            ypx_cnt = cv2.countNonZero(ymask[roi_y_slice, roi_x_slice])
            wpx_cnt = cv2.countNonZero(wmask[roi_y_slice, roi_x_slice])
            
            if (ypx_cnt > wpx_cnt) :
                stop = "YELLOW"
            else :
                stop = "WHITE"
                self.is_white = True
            
            stop_mask = np.zeros_like(lane_mask)
            # 좌표 변환 없이 직접 ROI 좌표에 그리기
            cv2.drawContours(stop_mask[y0:y1, :], [cnt], -1, 255, thickness=cv2.FILLED)
            break
        
        # 6. 차선 정제 (정지선 제거)
        clean_y = ymask.copy()
        clean_w = wmask.copy()
        
        if stop_mask is not None:
            clean_y &= ~stop_mask
            clean_w &= ~stop_mask
        
        # 상단 절단
        cut = int(self.lane_cut * h)
        kernel_3x3 = np.ones((3, 3), np.uint8)

        # 모폴로지 연산 (in-place)
        for mask in [clean_y, clean_w]:
            mask[cut:, :] = 0
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3x3, dst=mask)

        # 7. 차선 피팅
        y_lfit, y_rfit = self._fit(clean_y)
        w_lfit, w_rfit = self._fit(clean_w)

        # 8. 상태 및 우선순위에 따른 통합 피팅 결정
        lfit, rfit = None, None

        # 조건 1: 흰색 모드가 아니거나, 진행 방향이 LEFT인 경우 (노란색 우선)
        if (not self.is_white) or (self.mdir == "LEFT"):
            if y_lfit is not None or y_rfit is not None:
                lfit, rfit = y_lfit, y_rfit
            else:
                lfit, rfit = w_lfit, w_rfit
        # 조건 2: 그 외 상황 (흰색 우선)
        else:
            if w_lfit is not None or w_rfit is not None:
                lfit, rfit = w_lfit, w_rfit
            else:
                lfit, rfit = y_lfit, y_rfit

        combined_clean = cv2.bitwise_or(clean_y, clean_w)
        
        # 8. 디버깅 정보 생성
        ypx_roi = cv2.countNonZero(ymask[y0:y1, :])
        wpx_roi = cv2.countNonZero(wmask[y0:y1, :])
        debug_bev = self._debug_line(bev, combined_clean, lfit, rfit, stop, ypx_roi, wpx_roi, stop)
        
        if self.mstatus == "PARKING":
            return lfit, rfit, None, debug_bev
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

        # if self.mdir == "RIGHT":
        #     lx = self._base(hist, mid - 100, mid + 100, mid // 2)
        #     rx = self._base(hist, mid + 150, len(hist), mid + mid // 2)
        # elif self.mdir == "LEFT":
        #     lx = self._base(hist, 0, mid - 150, mid // 2)
        #     rx = self._base(hist, mid - 100, mid + 100, mid + mid // 2)
        # else:
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

    def _traffic_light(self, frame, debug_frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        roi = hsv[0:int(h * 0.4), int(w * 0.2):int(w * 0.8)]
        red = cv2.countNonZero(cv2.bitwise_or(cv2.inRange(roi, *self.RED1), cv2.inRange(roi, *self.RED2)))
        grn = cv2.countNonZero(cv2.inRange(roi, *self.GREEN))
        best = max((red, "RED"), (grn, "GREEN"))[1]
        light_result = best if max(red, grn) > 200 else "UNKNOWN"
        self.pub_tl.publish(light_result)

        # ==============DEBUG==============
        if light_result != "UNKNOWN":
            self._debug_light(debug_frame, 0, int(h * 0.4), int(w * 0.2), int(w * 0.8), light_result)
        return debug_frame

    def _ar_tag(self, frame, debug_frame):

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
            out += [float(ids[i][0]), dist, ang] # 오류나면 [0] 없애기
        if out:
            self.pub_ar.publish(Float32MultiArray(data=out))
        return debug_frame

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    CameraPerception().run()
