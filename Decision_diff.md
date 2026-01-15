--- Decision.py
+++ Decision_conservative_patched.py
@@ -227,7 +227,13 @@
         self.accum_error = 0
         self.prev_error = 0
         self.prev_steer = None
-        
+
+        # lane gate / ghost width tracking
+        # 왜: 한 프레임 오탐(폭 붕괴/급변) 또는 "한쪽 차선만 인식" 시, 조향이 크게 튀어 이탈을 유발.
+        # 기대효과: (1) 폭/점프 이상치 프레임 차단 + (2) 최근 정상 폭 기반 ghost 생성 + (3) 조향 급변 제한
+        self._last_lane_width_px = float(getattr(self.cfg, "lane_width_px_init", 342.0))
+        self._last_good_lane: Optional[Lane] = None
+        self._last_good_width_px: float = self._last_lane_width_px
+        self._last_good_t: float = 0.0
+        self._prev_cmd_steer: Optional[int] = None
         # --- config ---
         self.cfg = type("Cfg", (), {
             # steering / speed
@@ -256,6 +262,18 @@
             "lane_eval_y": float(gp("lane_eval_y", -1.0)),
             "obs_eval_x": float(gp("obs_eval_x", 0.5)),
 
+            # lane sanity gate (conservative safety net)
+            # 왜: 영상에서 이탈은 대부분 "한 프레임" 잘못 잡혀 폭이 붕괴하거나 폭이 순간 점프하는 시점에 발생.
+            # 기대효과: 폭이 말이 안 되는 프레임은 버리고, 직전 정상 차선을 짧게 유지하여 급조향/이탈을 방지.
+            "lane_gate_enable": bool(gp("lane_gate_enable", True)),
+            "lane_width_px_init": float(gp("lane_width_px_init", 342.0)),  # ~0.4m baseline
+            "lane_width_min_px": float(gp("lane_width_min_px", 240.0)),
+            "lane_width_max_px": float(gp("lane_width_max_px", 445.0)),
+            "lane_width_jump_max_px": float(gp("lane_width_jump_max_px", 90.0)),
+            "lane_hold_sec": float(gp("lane_hold_sec", 0.5)),
+
+            # steer slew-rate clamp (deg-like PWM step per loop)
+            # 왜: 오탐 프레임이 들어와도 조향이 한 번에 끝까지 튀지 않게 완충.
+            # 기대효과: "한 프레임 스파이크"를 실제 차량 조향으로 전달하기 전에 완만하게 만들어 트랙 이탈 확률 감소.
+            "steer_slew_enable": bool(gp("steer_slew_enable", True)),
+            "steer_slew_max_step": int(gp("steer_slew_max_step", 6)),
+
             # parking search (AR 없을 때)
             "park_search_sec": float(gp("park_search_sec", 1.0)),
             "park_search_speed": int(gp("park_search_speed", 98)),
@@ -327,24 +345,40 @@
             if tkey:
                 self._d[tkey] = now
 
+    
     def _cb_lane(self, m: Float32MultiArray) -> None:
         data = list(m.data) if m.data else []
         n = len(data)
-        y_eval = 470
-        LINE_WIDTH_PX = 400  # 실제 BEV상 차선 간격 px값에 맞춰 조정 필요
+        # 왜: 기존은 y_eval/center/폭이 하드코딩(470/320/400)이라
+        #     해상도/BEV 세팅이 조금만 바뀌면 "오른쪽/왼쪽 판별"과 ghost 폭이 틀어져 급조향을 유발.
+        # 기대효과: 화면 중심/폭을 동적으로 사용하고, ghost 폭을 최근 정상 폭으로 사용해 튐을 줄임.
+        y_eval = int(self.cfg.h * 0.8)
+        center_x = float(self.cfg.w) * 0.5
+        width_px = float(getattr(self, "_last_lane_width_px", float(self.cfg.lane_width_px_init)))
 
         if n >= 6:
             # 양쪽 차선 수신
             lane = Lane(data)
+            # 최근 정상 폭 업데이트 (양쪽이 있을 때만)
+            try:
+                w = self._lane_width_px(lane, y_eval)
+                if w == w and 1.0 <= w <= 5000.0:
+                    self._last_lane_width_px = float(w)
+            except Exception:
+                pass
 
         elif n >= 3:
-            # 한쪽 차선만 수신 (a, b, c 추출)
-            a, b, c = data[0], data[1], data[2]
-            target_x = a * (y_eval**2) + b * y_eval + c
-            if target_x > 320: # 감지된 것이 오른쪽 차선일 때
-                lane = Lane([a, b, c - LINE_WIDTH_PX, a, b, c])
-            else: # 감지된 것이 왼쪽 차선일 때
-                lane = Lane([a, b, c, a, b, c + LINE_WIDTH_PX])
-        else:
-            lane = None
+            a, b, c = float(data[0]), float(data[1]), float(data[2])
+            target_x = a * (y_eval ** 2) + b * y_eval + c
+
+            # 감지된 곡선이 오른쪽/왼쪽 중 어디로 더 가까운지로 판별 (320 하드코딩 제거)
+            # why: 혼합/점선에서 중앙 점선이 들어오면 320 기준 판별이 쉽게 틀어짐.
+            # expect: 해상도/BEV 변화에도 "오른쪽/왼쪽" 판별이 안정.
+            if target_x > center_x:  # right lane curve
+                lane = Lane([a, b, c - width_px, a, b, c])
+            else:  # left lane curve
+                lane = Lane([a, b, c, a, b, c + width_px])
+
         self._up("lane", lane)
 
@@ -372,11 +406,53 @@
                 ar=self._d["ar"],
             )
 
+    # ---------------- Lane filtering (conservative safety net) ----------------
+    def _lane_width_px(self, lane: "Lane", y_eval: int) -> float:
+        """Compute lane width at y_eval. Returns NaN if not computable."""
+        try:
+            xl = float(lane.left_x(y_eval))
+            xr = float(lane.right_x(y_eval))
+            return abs(xr - xl)
+        except Exception:
+            return float("nan")
+
+    def _filter_lane_for_drive(self, lane: Optional["Lane"], now_t: float) -> Optional["Lane"]:
+        """Conservative gate: reject suspicious lane frames and hold last good briefly."""
+        if not bool(getattr(self.cfg, "lane_gate_enable", True)):
+            return lane
+        if lane is None:
+            # hold if recent
+            if self._last_good_lane is not None and (now_t - self._last_good_t) <= float(self.cfg.lane_hold_sec):
+                return self._last_good_lane
+            return None
+
+        y_eval = int(self.cfg.h * 0.8)  # stable evaluation point
+        w = self._lane_width_px(lane, y_eval)
+
+        wmin = float(self.cfg.lane_width_min_px)
+        wmax = float(self.cfg.lane_width_max_px)
+        wjump = float(self.cfg.lane_width_jump_max_px)
+
+        # 왜: 차선 폭이 말이 안 되거나(붕괴/폭발) 직전 대비 급변하면 "그 프레임"은 거의 오탐.
+        # 기대효과: 오탐 프레임이 조향으로 직결되는 것을 차단하고, last_good을 짧게 유지해 차가 안정적으로 버팀.
+        ok = (w == w) and (wmin <= w <= wmax) and (abs(w - float(self._last_good_width_px)) <= wjump)
+
+        if ok:
+            self._last_good_lane = lane
+            self._last_good_width_px = w
+            self._last_good_t = now_t
+            return lane
+
+        # suspicious frame -> fall back to last good if fresh
+        if self._last_good_lane is not None and (now_t - self._last_good_t) <= float(self.cfg.lane_hold_sec):
+            return self._last_good_lane
+
+        return None
+
     # ---------------- Main loop ----------------
     def run(self) -> None:
         rate = rospy.Rate(30)
         while not rospy.is_shutdown():
             s = self._capture()
+            # conservative lane filter: block sudden lane-width glitches
+            s.lane = self._filter_lane_for_drive(s.lane, s.t)
             prev_fsm_state = self.fsm.state
             f = self.fsm.step(s)
@@
+        # conservative slew-rate clamp to prevent one-frame spikes
+        if bool(getattr(self.cfg, "steer_slew_enable", True)):
+            max_step = int(getattr(self.cfg, "steer_slew_max_step", 0))
+            if max_step > 0 and self._prev_cmd_steer is not None:
+                d = int(steer) - int(self._prev_cmd_steer)
+                if d > max_step:
+                    steer = int(self._prev_cmd_steer) + max_step
+                elif d < -max_step:
+                    steer = int(self._prev_cmd_steer) - max_step
+        self._prev_cmd_steer = int(steer)
+
         speed = int(speed)
 
         # motor
