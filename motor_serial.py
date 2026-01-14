#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_serial_v1_patched.py

역할:
- ROS 토픽(Int16MultiArray: [steer_pwm, speed_pwm])을 구독
- 아두이노(또는 모터 컨트롤러) 시리얼로 패킷 송신
- 토픽/시리얼 설정은 파라미터(~motor_topic, ~port, ~baud) 또는 YAML(~config)로 제어

v1 토픽 계약과의 호환:
- 기본 motor_topic: "cmd/motor" (상대 토픽)
- launch에서 <group ns="orda">로 묶으면 /orda/cmd/motor 가 됩니다.
- 기존 "/motor"와도 호환: _motor_topic:=/motor 로 실행하면 됩니다.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import rospy
import serial
import yaml
from std_msgs.msg import Int16MultiArray


# ---------------------------
# Helpers
# ---------------------------
def _safe_load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        rospy.logwarn(f"[MotorSerial] config not found: {path} (ignore)")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        rospy.logwarn(f"[MotorSerial] failed to load config: {path} ({e})")
        return {}


def _cfg_i(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _cfg_s(cfg: Dict[str, Any], key: str, default: str) -> str:
    v = cfg.get(key, default)
    return default if v is None else str(v)


# ---------------------------
# Config
# ---------------------------
@dataclass
class MotorSerialConfig:
    motor_topic: str = "cmd/motor"       # v1 기본
    port: str = "/dev/arduino"
    baud: int = 115200
    timeout_sec: float = 1.0

    # PWM 안전 범위 (기본값은 기존 코드 유지)
    steer_min: int = 45
    steer_max: int = 135
    speed_min: int = 0
    speed_max: int = 150

    # 종료/비상 시 보낼 중립 값
    neutral_steer: int = 90
    neutral_speed: int = 90

    # 호환 옵션: speed=0을 '정지(중립 PWM)'로 간주
    zero_speed_means_neutral: bool = True

    # 시리얼 연결 재시도
    open_retry_count: int = 50
    open_retry_sleep_sec: float = 0.1

    @staticmethod
    def from_sources(yaml_cfg: Dict[str, Any]) -> "MotorSerialConfig":
        """
        우선순위:
        1) ROS private param (~motor_topic, ~port, ~baud, ...)
        2) YAML (~config) 내 key
        3) 기본값
        """
        c = MotorSerialConfig()

        # YAML → 기본값 덮어쓰기 (ROS param이 있으면 이후에 다시 덮어씀)
        if yaml_cfg:
            c.motor_topic = _cfg_s(yaml_cfg, "motor_topic", c.motor_topic)
            c.port = _cfg_s(yaml_cfg, "motor_serial_port", _cfg_s(yaml_cfg, "port", c.port))
            c.baud = _cfg_i(yaml_cfg, "motor_serial_baud", _cfg_i(yaml_cfg, "baud", c.baud))

            c.steer_min = _cfg_i(yaml_cfg, "steer_min", c.steer_min)
            c.steer_max = _cfg_i(yaml_cfg, "steer_max", c.steer_max)
            c.speed_min = _cfg_i(yaml_cfg, "speed_min", c.speed_min)
            c.speed_max = _cfg_i(yaml_cfg, "speed_max", c.speed_max)

            c.neutral_steer = _cfg_i(yaml_cfg, "steer_center", c.neutral_steer)  # runtime과 맞추기
            c.neutral_speed = _cfg_i(yaml_cfg, "pwm_stop", c.neutral_speed)

        # ROS param 최우선
        c.motor_topic = rospy.get_param("~motor_topic", c.motor_topic)
        c.port = rospy.get_param("~port", c.port)
        c.baud = int(rospy.get_param("~baud", c.baud))
        c.timeout_sec = float(rospy.get_param("~timeout_sec", c.timeout_sec))

        c.steer_min = int(rospy.get_param("~steer_min", c.steer_min))
        c.steer_max = int(rospy.get_param("~steer_max", c.steer_max))
        c.speed_min = int(rospy.get_param("~speed_min", c.speed_min))
        c.speed_max = int(rospy.get_param("~speed_max", c.speed_max))

        c.neutral_steer = int(rospy.get_param("~neutral_steer", c.neutral_steer))
        c.neutral_speed = int(rospy.get_param("~neutral_speed", c.neutral_speed))
        c.zero_speed_means_neutral = bool(int(rospy.get_param("~zero_speed_means_neutral", int(c.zero_speed_means_neutral))))

        c.open_retry_count = int(rospy.get_param("~open_retry_count", c.open_retry_count))
        c.open_retry_sleep_sec = float(rospy.get_param("~open_retry_sleep_sec", c.open_retry_sleep_sec))

        return c


# ---------------------------
# Serial bridge
# ---------------------------
class MotorSerialBridge:
    def __init__(self, cfg: MotorSerialConfig):
        self.cfg = cfg
        self.ser: Optional[serial.Serial] = None

    def open(self) -> None:
        last_err: Optional[Exception] = None
        for i in range(max(1, self.cfg.open_retry_count)):
            try:
                self.ser = serial.Serial(
                    self.cfg.port,
                    self.cfg.baud,
                    timeout=self.cfg.timeout_sec,
                )
                rospy.loginfo(f"[MotorSerial] serial opened: {self.cfg.port} @ {self.cfg.baud}")
                return
            except Exception as e:
                last_err = e
                rospy.logwarn(f"[MotorSerial] serial open failed ({i+1}/{self.cfg.open_retry_count}): {e}")
                time.sleep(self.cfg.open_retry_sleep_sec)

        raise RuntimeError(f"serial open failed: {self.cfg.port} ({last_err})")

    def close(self) -> None:
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _clamp(self, v: int, lo: int, hi: int) -> int:
        return lo if v < lo else (hi if v > hi else v)

    def create_command(self, steer: int, speed: int) -> bytearray:
        # 프로토콜은 기존 코드 유지
        STX = 0xEA
        ETX = 0x03
        Length = 0x03
        d1 = 0
        d2 = 0

        steer = self._clamp(int(steer), self.cfg.steer_min, self.cfg.steer_max)
        # speed 계약 호환: 다른 노드가 0을 보내면 중립 PWM으로 변환
        if getattr(self.cfg, 'zero_speed_means_neutral', True) and int(speed) == 0:
            speed = int(self.cfg.neutral_speed)

        speed = self._clamp(int(speed), self.cfg.speed_min, self.cfg.speed_max)

        # 체크섬 계산
        cs = ((~(Length + steer + speed + d1 + d2)) & 0xFF) + 1
        return bytearray([STX, Length, steer, speed, d1, d2, cs, ETX])

    def send(self, steer: int, speed: int) -> None:
        if self.ser is None:
            return
        pkt = self.create_command(steer, speed)
        self.ser.write(pkt)

    def send_neutral(self) -> None:
        try:
            self.send(self.cfg.neutral_steer, self.cfg.neutral_speed)
        except Exception:
            pass


# ---------------------------
# ROS Node
# ---------------------------
class MotorSerialNode:
    def __init__(self):
        rospy.init_node("motor_serial_node", anonymous=False)

        # YAML(선택) + ROS param 조합
        yaml_path = rospy.get_param("~config", "")
        yaml_cfg = _safe_load_yaml(yaml_path)
        self.cfg = MotorSerialConfig.from_sources(yaml_cfg)

        # Serial
        self.bridge = MotorSerialBridge(self.cfg)
        try:
            self.bridge.open()
        except Exception as e:
            rospy.logerr(f"[MotorSerial] cannot open serial: {e}")
            raise

        # Subscriber (topic hardcode 제거)
        rospy.Subscriber(self.cfg.motor_topic, Int16MultiArray, self._on_motor, queue_size=1)

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[MotorSerial] ready")
        rospy.loginfo(f"[MotorSerial] motor_topic: {self.cfg.motor_topic}")
        rospy.loginfo(f"[MotorSerial] neutral: steer={self.cfg.neutral_steer}, speed={self.cfg.neutral_speed}")

    def _on_motor(self, msg: Int16MultiArray) -> None:
        try:
            if not msg.data or len(msg.data) < 2:
                return
            steer = int(msg.data[0])
            speed = int(msg.data[1])
            self.bridge.send(steer, speed)
        except Exception as e:
            rospy.logerr(f"[MotorSerial] send error: {e}")

    def _on_shutdown(self) -> None:
        rospy.loginfo("[MotorSerial] shutdown: sending neutral + closing serial")
        try:
            self.bridge.send_neutral()
        finally:
            self.bridge.close()

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    node = MotorSerialNode()
    node.spin()


if __name__ == "__main__":
    main()
