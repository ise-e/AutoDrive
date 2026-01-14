#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_serial.py (multi-topic patched)

역할:
- ROS 토픽(std_msgs/Int16MultiArray: [steer_pwm, speed_pwm])을 구독
- 아두이노(또는 모터 컨트롤러) 시리얼로 패킷 송신

이번 패치 핵심:
- Decision이 /cmd/motor 를 내보내는데 motor_serial이 /motor 를 보고 있거나(launch/param) 그 반대인 경우가 많아서,
  기본으로 두 토픽을 **동시에 구독**하도록 바꿨습니다.
  - 기본 구독: /cmd/motor, /motor
  - 호환: ~motor_topic (단일) / ~motor_topics (복수) 지원
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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
        if not isinstance(data, dict):
            rospy.logwarn(f"[MotorSerial] config is not a dict: {path} (ignore)")
            return {}
        return data
    except Exception as e:
        rospy.logwarn(f"[MotorSerial] failed to read config {path}: {e} (ignore)")
        return {}


def _cfg_i(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _cfg_f(cfg: Dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _cfg_s(cfg: Dict[str, Any], key: str, default: str) -> str:
    v = cfg.get(key, default)
    return default if v is None else str(v)


def _cfg_topics(cfg: Dict[str, Any], key: str, default: Sequence[str]) -> List[str]:
    v = cfg.get(key, None)
    if v is None:
        return list(default)
    # yaml list
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out if out else list(default)
    # comma separated string
    s = str(v).strip()
    if not s:
        return list(default)
    return [t.strip() for t in s.split(",") if t.strip()]


def _dedup(seq: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------
# Config
# ---------------------------
@dataclass
class MotorSerialConfig:
    # 기본은 "둘 다" 구독 (현장 호환성 최우선)
    motor_topics: List[str] = None  # type: ignore

    # legacy 단일 토픽 파라미터(launch에서 쓰던 경우)
    motor_topic: str = "cmd/motor"

    port: str = "/dev/arduino"
    baud: int = 115200
    timeout_sec: float = 1.0

    # PWM 안전 범위
    steer_min: int = 45
    steer_max: int = 135
    speed_min: int = 0
    speed_max: int = 150

    # 중립값
    neutral_steer: int = 90
    neutral_speed: int = 90

    # speed=0을 "중립"으로 취급할지(기존 코드 호환)
    zero_speed_means_neutral: bool = True

    # 시리얼 open 재시도
    open_retry_count: int = 30
    open_retry_sleep_sec: float = 0.1

    def __post_init__(self) -> None:
        if self.motor_topics is None:
            # 기본값: 두 토픽 모두 구독
            self.motor_topics = ["/cmd/motor", "/motor"]

    @staticmethod
    def from_sources(yaml_cfg: Dict[str, Any]) -> "MotorSerialConfig":
        c = MotorSerialConfig()

        # YAML 우선 (있으면)
        if yaml_cfg:
            c.motor_topic = _cfg_s(yaml_cfg, "motor_topic", c.motor_topic)
            c.motor_topics = _cfg_topics(yaml_cfg, "motor_topics", c.motor_topics)

            c.port = _cfg_s(yaml_cfg, "port", c.port)
            c.baud = _cfg_i(yaml_cfg, "baud", c.baud)
            c.timeout_sec = _cfg_f(yaml_cfg, "timeout_sec", c.timeout_sec)

            c.steer_min = _cfg_i(yaml_cfg, "steer_min", c.steer_min)
            c.steer_max = _cfg_i(yaml_cfg, "steer_max", c.steer_max)
            c.speed_min = _cfg_i(yaml_cfg, "speed_min", c.speed_min)
            c.speed_max = _cfg_i(yaml_cfg, "speed_max", c.speed_max)

            # 다른 코드에서 쓰던 키 호환
            c.neutral_steer = _cfg_i(yaml_cfg, "steer_center", c.neutral_steer)
            c.neutral_speed = _cfg_i(yaml_cfg, "pwm_stop", c.neutral_speed)

        # ROS param 최우선
        # - ~motor_topics: list or "a,b,c"
        # - ~motor_topic: legacy 단일
        ros_topics = rospy.get_param("~motor_topics", None)
        if ros_topics is not None:
            # rospy가 list로 주거나, 문자열로 주거나 둘 다 가능
            if isinstance(ros_topics, (list, tuple)):
                c.motor_topics = [str(x).strip() for x in ros_topics if str(x).strip()]
            else:
                c.motor_topics = [t.strip() for t in str(ros_topics).split(",") if t.strip()]

        c.motor_topic = rospy.get_param("~motor_topic", c.motor_topic)

        # motor_topics가 비어있으면 legacy 단일로 대체
        if not c.motor_topics:
            c.motor_topics = [c.motor_topic]

        # legacy 단일을 motor_topics에도 포함시키되 중복 제거
        c.motor_topics = _dedup([c.motor_topic] + list(c.motor_topics))

        c.port = rospy.get_param("~port", c.port)
        c.baud = int(rospy.get_param("~baud", c.baud))
        c.timeout_sec = float(rospy.get_param("~timeout_sec", c.timeout_sec))

        c.steer_min = int(rospy.get_param("~steer_min", c.steer_min))
        c.steer_max = int(rospy.get_param("~steer_max", c.steer_max))
        c.speed_min = int(rospy.get_param("~speed_min", c.speed_min))
        c.speed_max = int(rospy.get_param("~speed_max", c.speed_max))

        c.neutral_steer = int(rospy.get_param("~steer_center", c.neutral_steer))
        c.neutral_speed = int(rospy.get_param("~pwm_stop", c.neutral_speed))

        c.zero_speed_means_neutral = bool(int(rospy.get_param("~zero_speed_means_neutral", int(c.zero_speed_means_neutral))))

        c.open_retry_count = int(rospy.get_param("~open_retry_count", c.open_retry_count))
        c.open_retry_sleep_sec = float(rospy.get_param("~open_retry_sleep_sec", c.open_retry_sleep_sec))

        return c


# ---------------------------
# Serial bridge
# ---------------------------
class MotorSerialBridge:
    """
    기존 motor_serial 패킷 포맷 유지:
    STX, Length, steer, speed, d1, d2, checksum, ETX
    """
    STX = 0x53
    ETX = 0x45
    LENGTH = 0x05
    D1 = 0x00
    D2 = 0x00

    def __init__(self, cfg: MotorSerialConfig):
        self.cfg = cfg
        self.ser: Optional[serial.Serial] = None

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return lo if v < lo else hi if v > hi else v

    def open(self) -> None:
        last_err: Optional[Exception] = None
        for _ in range(max(1, self.cfg.open_retry_count)):
            try:
                self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_sec)
                rospy.loginfo(f"[MotorSerial] serial opened: {self.cfg.port} @ {self.cfg.baud}")
                return
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.open_retry_sleep_sec)
        raise RuntimeError(f"cannot open serial {self.cfg.port}: {last_err}")

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def create_command(self, steer: int, speed: int) -> bytearray:
        steer = self._clamp(int(steer), self.cfg.steer_min, self.cfg.steer_max)

        if getattr(self.cfg, "zero_speed_means_neutral", True) and int(speed) == 0:
            speed = int(self.cfg.neutral_speed)
        speed = self._clamp(int(speed), self.cfg.speed_min, self.cfg.speed_max)

        length = self.LENGTH
        d1 = self.D1
        d2 = self.D2
        cs = ((~(length + steer + speed + d1 + d2)) & 0xFF) + 1  # 8-bit 2's complement
        return bytearray([self.STX, length, steer & 0xFF, speed & 0xFF, d1, d2, cs & 0xFF, self.ETX])

    def send(self, steer: int, speed: int) -> None:
        if self.ser is None:
            return
        self.ser.write(self.create_command(steer, speed))

    def send_neutral(self) -> None:
        self.send(int(self.cfg.neutral_steer), int(self.cfg.neutral_speed))


# ---------------------------
# ROS Node
# ---------------------------
class MotorSerialNode:
    def __init__(self):
        rospy.init_node("motor_serial_node", anonymous=False)

        yaml_path = rospy.get_param("~config", "")
        yaml_cfg = _safe_load_yaml(yaml_path)
        self.cfg = MotorSerialConfig.from_sources(yaml_cfg)

        self.bridge = MotorSerialBridge(self.cfg)
        self.bridge.open()

        # 구독 토픽(여러개)
        self._subs = []
        for t in self.cfg.motor_topics:
            # resolve_name은 상대/절대 토픽을 실제 네임스페이스로 변환해줌
            resolved = rospy.resolve_name(t)
            self._subs.append(
                rospy.Subscriber(resolved, Int16MultiArray, self._on_motor, queue_size=1)
            )

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[MotorSerial] ready")
        rospy.loginfo(f"[MotorSerial] subscribing: {', '.join([rospy.resolve_name(t) for t in self.cfg.motor_topics])}")
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
    MotorSerialNode().spin()


if __name__ == "__main__":
    main()
