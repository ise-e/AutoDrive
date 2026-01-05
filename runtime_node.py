#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Int16MultiArray
from Decision import DecisionNode

def main():
    rospy.init_node("runtime_orchestrator")
    
    # 클래스 생성 (구독자 자동 연결됨)
    node = DecisionNode()
    
    motor_pub = rospy.Publisher("/motor", Int16MultiArray, queue_size=1)
    
    # 30Hz 제어 루프
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        # 1회 스텝 실행 후 결과 반환
        steer, speed = node.step()
        
        # 모터 명령 발행
        motor_pub.publish(Int16MultiArray(data=[steer, speed]))
        
        rate.sleep()

if __name__ == '__main__':
    main()