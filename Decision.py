
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
            speed_caution=int(rospy.get_param("~speed_caution", 95)),
            speed_parking=int(rospy.get_param("~speed_parking", 98)),
            speed_min=int(rospy.get_param("~speed_min", 0)),
            creep_enable=bool(rospy.get_param("~creep_enable", True)),
            speed_creep=int(rospy.get_param("~speed_creep", 99)),
            creep_max_sec=float(rospy.get_param("~creep_max_sec", 6.0)),
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
    def _stop_wait_creep_allowed(self, s: 'WorldState', now: float) -> bool:
        """CHECK_STOP 구간에서 '정지선 확정(YELLOW)'이 아직 안 된 동안만 저속 크립 주행."""
        if not self.cfg.creep_enable:
            return False

        # 어떤 CHECK_STOP 상태인지 확인
        st_name = getattr(self.mission.state, "name", "")
        if "CHECK_STOP" not in st_name:
            return False

        # 정지선이 확정으로 잡히면 즉시 정지(신호 대기)
        if getattr(s, "stop_sign", "NONE") == "YELLOW":
            return False

        # 너무 오래 크립하면 위험 -> 정지
        enter_t = float(getattr(self.mission, "last_state_change", now))
        if (now - enter_t) > float(self.cfg.creep_max_sec):
            return False

        # 최소한 한쪽 차선센서가 살아있어야 크립 허용 (조향 불가하면 위험)
        cam_ok = (s.cam is not None) and (len(getattr(s.cam, "coeffs", []) or []) >= 3)
        lidar_ok = (s.lidar is not None) and (len(getattr(s.lidar, "coeffs", []) or []) >= 3)
        return cam_ok or lidar_ok

    def _compute_fused_error(self, s: 'WorldState', now: float):
        """현재 로직과 동일한 방식으로 (최종 정규화 에러, 유효 여부) 반환."""
        cam_err = s.cam.get_error_norm(self.cfg) if s.cam else 0.0
        lidar_err = s.lidar.get_error_norm(self.cfg) if s.lidar else 0.0

        cam_age = (now - s.cam.timestamp) if s.cam else 1e9
        lidar_age = (now - s.lidar.timestamp) if s.lidar else 1e9

        cam_f = self._fresh_weight(cam_age, self.cfg.watchdog_hard_sec)
        lidar_f = self._fresh_weight(lidar_age, self.cfg.watchdog_hard_sec)

        cam_n = len(getattr(s.cam, "coeffs", []) or []) if s.cam else 0
        lidar_n = len(getattr(s.lidar, "coeffs", []) or []) if s.lidar else 0

        cam_r = float(self.cfg.cam_double_lane_weight) if cam_n >= 6 else (float(self.cfg.cam_single_lane_weight) if cam_n >= 3 else 0.0)
        lidar_r = float(self.cfg.lidar_double_lane_weight) if lidar_n >= 6 else (float(self.cfg.lidar_single_lane_weight) if lidar_n >= 3 else 0.0)

        w_cam = cam_r * cam_f
        w_lidar = lidar_r * lidar_f

        w_sum = w_cam + w_lidar
        if w_sum <= 1e-6:
            return 0.0, False

        final_err = (w_cam * cam_err + w_lidar * lidar_err) / w_sum
        return final_err, True


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

            if status == "FINISH":
                steer = self.cfg.steer_center
                speed = 0

            elif status == "STOP":
                # 기본은 정지. 단, CHECK_STOP 구간에서 정지선/신호등이 아직 안 잡힐 때는 저속 크립 주행 허용.
                if self._stop_wait_creep_allowed(s, now):
                    final_err, is_valid = self._compute_fused_error(s, now)
                    if is_valid:
                        steer_cmd = self.ctrl.step(final_err, dt)
                        steer = self.cfg.steer_center - int(steer_cmd)
                        speed = self.cfg.speed_creep
                    else:
                        steer = self.cfg.steer_center
                        speed = 0
                else:
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