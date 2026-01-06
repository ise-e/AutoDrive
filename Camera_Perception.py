#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray, String

def now_sec():
    return time.time()

class CameraPerception:
    def __init__(self):
        rospy.init_node("camera_perception")

        self._load_params()
        self.mdir, self.mstatus = "RIGHT", "NONE"
        self.last_fit, self.last_fit_t = None, 0.0

        self.pub_lane = rospy.Publisher("/lane_coeffs", Float32MultiArray, queue_size=1)
        self.pub_stop = rospy.Publisher("/stop_line_status", String, queue_size=1)
        self.pub_tl = rospy.Publisher("/traffic_light_status", String, queue_size=1)
        self.pub_ar = rospy.Publisher("/ar_tag_info", Float32MultiArray, queue_size=1)
        self.pub_ov = rospy.Publisher(self.overlay_topic, CompressedImage, queue_size=1)

        rospy.Subscriber(self.camera_topic, Image, self._on_img, queue_size=1)
        rospy.Subscriber("/mission_direction", String, lambda m: setattr(self, "mdir", (m.data or "RIGHT").upper()), queue_size=1)
        rospy.Subscriber("/mission_status", String, lambda m: setattr(self, "mstatus", (m.data or "NONE").upper()), queue_size=1)

    def _load_params(self):
        p = rospy.get_param
        d = {
            "camera_topic": "/camera/image_raw",
            "out_w": 640, "out_h": 480,
            "overlay_topic": "/debug/lane_overlay/image/compressed",
            "overlay_enable": True, "jpeg_quality": 80,
            "src_pts_ratio": [200/640, 300/480, 440/640, 300/480, 50/640, 450/480, 590/640, 450/480],
            "dst_pts_ratio": [150/640, 0.0, 490/640, 0.0, 150/640, 1.0, 490/640, 1.0],
            "nwindows": 9, "margin": 60, "minpix": 50, "min_points": 220,
            "stop_check_y0": 0.70, "stop_check_y1": 0.80, "stop_px_per_col": 15.0, "lane_fit_ymax": 0.70,
            "last_fit_hold_sec": 0.5,
            "tag_size": 0.15, "focal_length": 600.0,
        }
        for k, v in d.items():
            setattr(self, k, p("~" + k, v))

        # HSV thresholds
        self.YELLOW = (np.array([15, 80, 80]), np.array([40, 255, 255]))
        self.WHITE = (np.array([0, 0, 200]), np.array([180, 40, 255]))
        self.RED1 = (np.array([0, 80, 80]), np.array([10, 255, 255]))
        self.RED2 = (np.array([160, 80, 80]), np.array([179, 255, 255]))
        self.GREEN = (np.array([40, 80, 80]), np.array([90, 255, 255]))

        w, h = float(self.out_w), float(self.out_h)
        sp = np.array([[self.src_pts_ratio[i]*w, self.src_pts_ratio[i+1]*h] for i in range(0, 8, 2)], np.float32)
        dp = np.array([[self.dst_pts_ratio[i]*w, self.dst_pts_ratio[i+1]*h] for i in range(0, 8, 2)], np.float32)
        self.M = cv2.getPerspectiveTransform(sp, dp)
        self.Minv = cv2.getPerspectiveTransform(dp, sp)

    def _rosimg_to_bgr(self, msg):
        if msg.encoding == "bgr8":
            return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return None

    def _publish_overlay(self, bgr):
        if (not bool(self.overlay_enable)) or self.pub_ov.get_num_connections() <= 0:
            return
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if ok:
            self.pub_ov.publish(CompressedImage(format="jpeg", data=enc.tobytes()))

    def _fit(self, binimg):
        hist = np.sum(binimg[binimg.shape[0] // 2:, :], axis=0)
        mid = len(hist) // 2

        def base(a, b, fallback):
            seg = hist[max(0, int(a)):max(0, int(b))]
            return int(np.argmax(seg) + int(a)) if seg.size else int(fallback)

        if self.mdir == "RIGHT":
            lx, rx = base(mid - 100, mid + 100, mid // 2), base(mid + 150, len(hist), mid + mid // 2)
        elif self.mdir == "LEFT":
            lx, rx = base(0, mid - 150, mid // 2), base(mid - 100, mid + 100, mid + mid // 2)
        else:
            lx, rx = base(0, mid, mid // 2), base(mid, len(hist), mid + mid // 2)

        nw = max(3, int(self.nwindows))
        win_h, margin, minpix = binimg.shape[0] // nw, int(self.margin), int(self.minpix)
        nzy, nzx = np.nonzero(binimg)
        lidx, ridx = [], []

        for win in range(nw):
            y0, y1 = binimg.shape[0] - (win + 1) * win_h, binimg.shape[0] - win * win_h
            gl = ((nzy >= y0) & (nzy < y1) & (nzx >= lx - margin) & (nzx < lx + margin)).nonzero()[0]
            gr = ((nzy >= y0) & (nzy < y1) & (nzx >= rx - margin) & (nzx < rx + margin)).nonzero()[0]
            lidx.append(gl); ridx.append(gr)
            if len(gl) > minpix: lx = int(np.mean(nzx[gl]))
            if len(gr) > minpix: rx = int(np.mean(nzx[gr]))

        li = np.concatenate(lidx) if lidx else np.array([], np.int32)
        ri = np.concatenate(ridx) if ridx else np.array([], np.int32)
        if len(li) < int(self.min_points) or len(ri) < int(self.min_points):
            return None, None
        return np.polyfit(nzy[li], nzx[li], 2), np.polyfit(nzy[ri], nzx[ri], 2)
def _debug(self, frame, bev, lane, lfit, rfit, stop, stop_roi):
    # frame: 원본 시점(카메라) 영상(리사이즈된 out_w/out_h)
    # bev  : 차선 검출에 사용한 BEV 영상(디버그용이 아니라 변환 좌표계용)
    # lane : BEV에서 만든 바이너리 마스크
    dbg = frame.copy()
    h, w = lane.shape[:2]

    # ----- 1) 차선(중앙선/좌/우) 시각화: BEV 좌표 -> 카메라 좌표로 역투영 -----
    if lfit is not None and rfit is not None:
        ys = np.linspace(0, h - 1, 25, dtype=np.float32)
        xl = np.polyval(lfit, ys).astype(np.float32)
        xr = np.polyval(rfit, ys).astype(np.float32)
        cx = (xl + xr) * 0.5

        pts_c_bev = np.stack([cx, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
        pts_l_bev = np.stack([xl, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
        pts_r_bev = np.stack([xr, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)

        pts_c = cv2.perspectiveTransform(pts_c_bev, self.Minv).astype(np.int32)
        pts_l = cv2.perspectiveTransform(pts_l_bev, self.Minv).astype(np.int32)
        pts_r = cv2.perspectiveTransform(pts_r_bev, self.Minv).astype(np.int32)

        cv2.polylines(dbg, [pts_l], False, (255, 0, 0), 2)   # left
        cv2.polylines(dbg, [pts_r], False, (0, 0, 255), 2)   # right
        cv2.polylines(dbg, [pts_c], False, (0, 255, 0), 2)   # center

    # ----- 2) 정지선 체크 ROI 시각화 (BEV ROI를 카메라로 역투영해서 폴리곤 표시) -----
    y0, y1 = stop_roi
    roi_bev = np.array([
        [[0, y0]],
        [[w - 1, y0]],
        [[w - 1, y1]],
        [[0, y1]],
    ], dtype=np.float32)
    roi_img = cv2.perspectiveTransform(roi_bev, self.Minv).astype(np.int32)

    # stop 상태에 따라 색만 다르게 표시
    roi_color = (0, 0, 255) if stop != "NONE" else (0, 255, 255)
    cv2.polylines(dbg, [roi_img], True, roi_color, 2)

    # ----- 3) 텍스트 오버레이 -----
    cv2.putText(
        dbg,
        f"STOP={stop}  mdir={self.mdir}  st={self.mstatus}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return dbg
def _lane_stop(self, frame, bev):
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    ymask = cv2.inRange(hsv, *self.YELLOW)
    wmask = cv2.inRange(hsv, *self.WHITE)

    h, w = ymask.shape
    y0 = int(float(self.stop_check_y0) * h)
    y1 = int(float(self.stop_check_y1) * h)
    y1 = min(h, max(y0 + 1, y1))

    ypx = cv2.countNonZero(ymask[y0:y1, :])
    wpx = cv2.countNonZero(wmask[y0:y1, :])

    stop = "YELLOW" if ypx > w * float(self.stop_px_per_col) else ("WHITE" if wpx > w * float(self.stop_px_per_col) else "NONE")

    lane = cv2.bitwise_or(ymask, wmask)
    lane[int(float(self.lane_fit_ymax) * h):, :] = 0
    lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    lfit, rfit = self._fit(lane)
    dbg = self._debug(frame, bev, lane, lfit, rfit, stop, (y0, y1))
    return lfit, rfit, stop, dbg
    def _traffic_light(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        roi = hsv[0:int(h * 0.4), int(w * 0.2):int(w * 0.8)]
        red = cv2.countNonZero(cv2.bitwise_or(cv2.inRange(roi, *self.RED1), cv2.inRange(roi, *self.RED2)))
        yel = cv2.countNonZero(cv2.inRange(roi, *self.YELLOW))
        grn = cv2.countNonZero(cv2.inRange(roi, *self.GREEN))
        m = max(red, yel, grn)
        self.pub_tl.publish(String(data=("RED" if m == red else ("YELLOW" if m == yel else "GREEN")) if m > 200 else "UNKNOWN"))

    def _ar_tag(self, frame):
        if not hasattr(cv2, "aruco"):
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ar = cv2.aruco
        dic = ar.Dictionary_get(ar.DICT_6X6_250)
        pars = ar.DetectorParameters_create()
        corners, ids, _ = ar.detectMarkers(gray, dic, parameters=pars)
        if ids is None or len(ids) == 0:
            return
        tag, focal, cx = float(self.tag_size), float(self.focal_length), gray.shape[1] * 0.5
        out = []
        for i in range(len(ids)):
            c = corners[i][0]
            pw = float(np.linalg.norm(c[0] - c[1]))
            if pw > 1e-6:
                dist = (tag * focal) / pw
                ang = float(np.arctan2(float(np.mean(c[:, 0]) - cx), focal))
                out += [float(ids[i]), dist, ang]
        if out:
            self.pub_ar.publish(Float32MultiArray(data=out))

    def _on_img(self, msg):
        frame = self._rosimg_to_bgr(msg)
        if frame is None:
            return

        frame = cv2.resize(frame, (int(self.out_w), int(self.out_h)))
        bev = cv2.warpPerspective(frame, self.M, (int(self.out_w), int(self.out_h)))

        lfit, rfit, stop, dbg = self._lane_stop(frame, bev)
        self.pub_stop.publish(String(data=stop))

        if lfit is not None and rfit is not None:
            self.last_fit, self.last_fit_t = (lfit, rfit), now_sec()
            self.pub_lane.publish(Float32MultiArray(data=list(lfit) + list(rfit)))
        elif self.last_fit and (now_sec() - self.last_fit_t) <= float(self.last_fit_hold_sec):
            lf, rf = self.last_fit
            self.pub_lane.publish(Float32MultiArray(data=list(lf) + list(rf)))
        else:
            self.pub_lane.publish(Float32MultiArray(data=[]))

        if self.mstatus == "STOP":
            self._traffic_light(frame)
        if self.mstatus == "PARKING":
            self._ar_tag(frame)

        self._publish_overlay(dbg)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    CameraPerception().run()
