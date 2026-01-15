--- Camera_Perception.py
+++ Camera_Perception_patched (2).py
@@ -326,28 +326,36 @@
             mask[cut:, :] = 0
             cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3x3, dst=mask)
 
-        # 7. 차선 피팅
+        # (추가) 차선 피팅에서는 정지선 ROI를 항상 제외 (정지선 미검출 프레임에서도 안전)
+        # 왜: 정지선은 "가로선"이라 히스토그램/슬라이딩윈도우에 강하게 걸려 차선으로 오인식될 수 있음.
+        #     기존에는 정지선 검출 성공 시에만 stop_mask로 제거되거나, 검출 실패 프레임에서는 그대로 남을 수 있었음.
+        # 기대효과: stopline 검출이 흔들려도(한 프레임 실패) 차선 피팅 입력에서 정지선이 섞이지 않아
+        #          "정지선=차선" 오인식으로 인한 급조향/이탈을 크게 줄임.
+        clean_y[y0:y1, :] = 0
+        clean_w[y0:y1, :] = 0
+
+        # 7. 차선 피팅 (색상 분리 결과 + 통합 결과)
         y_lfit, y_rfit = self._fit(clean_y)
         w_lfit, w_rfit = self._fit(clean_w)
 
-        # 8. 상태 및 우선순위에 따른 통합 피팅 결정
+        combined_clean = cv2.bitwise_or(clean_y, clean_w)
+        c_lfit, c_rfit = self._fit(combined_clean)
+
+        # 8. 최종 차선 선택: 통합 결과 우선, 실패 시 기존 우선순위로 fallback
+        # 왜: 혼합 차선(왼 노랑 / 오 흰)에서는 ymask/wmask 각각으로는 "반쪽짜리"가 되기 쉬움.
+        #     기존 로직은 "노랑 세트" 또는 "흰 세트" 중 하나만 선택하는 구조라 혼합구간에서 좌/우 동시 인식이 깨짐.
+        # 기대효과: 주행용 인식은 색상 구분을 버리고(색맹), 좌/우를 한 프레임에서 동시에 잡을 확률을 올려
+        #          ghost lane 의존도를 줄이고, 곡선+점선에서 바깥으로 튀는 빈도를 줄임.
         lfit, rfit = None, None
-
-        # 조건 1: 흰색 모드가 아니거나, 진행 방향이 LEFT인 경우 (노란색 우선)
-        if (not self.is_white) or (self.mdir == "LEFT"):
-            if y_lfit is not None or y_rfit is not None:
-                lfit, rfit = y_lfit, y_rfit
+        if (c_lfit is not None) or (c_rfit is not None):
+            lfit, rfit = c_lfit, c_rfit
+        else:
+            # 조건 1: 흰색 모드가 아니거나, 진행 방향이 LEFT인 경우 (노란색 우선)
+            if (not self.is_white) or (self.mdir == "LEFT"):
+                if y_lfit is not None or y_rfit is not None:
+                    lfit, rfit = y_lfit, y_rfit
+                else:
+                    lfit, rfit = w_lfit, w_rfit
+            # 조건 2: 그 외 상황 (흰색 우선)
             else:
-                lfit, rfit = w_lfit, w_rfit
-        # 조건 2: 그 외 상황 (흰색 우선)
-        else:
-            if w_lfit is not None or w_rfit is not None:
-                lfit, rfit = w_lfit, w_rfit
-            else:
-                lfit, rfit = y_lfit, y_rfit
-
-        combined_clean = cv2.bitwise_or(clean_y, clean_w)
-        
+                if w_lfit is not None or w_rfit is not None:
+                    lfit, rfit = w_lfit, w_rfit
+                else:
+                    lfit, rfit = y_lfit, y_rfit
+
         # 8. 디버깅 정보 생성
         ypx_roi = cv2.countNonZero(ymask[y0:y1, :])
         wpx_roi = cv2.countNonZero(wmask[y0:y1, :])
@@ -607,4 +615,4 @@
 
 
 if __name__ == "__main__":
-    CameraPerception().run()
+    CameraPerception().run()
