#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def start_node():
    rospy.init_node('img_publisher_node')
    # 카메라 이미지 토픽 발행
    pub = rospy.Publisher('camera/image_raw', Image, queue_size=10)
    
    bridge = CvBridge()
    rate = rospy.Rate(30) # 30Hz

    # ------------------------------------------------------
    # [수정] launch 파일에서 'device_id' 파라미터를 받아옵니다.
    # 기본값은 0입니다 (/dev/video0).
    # ------------------------------------------------------
    device_id = rospy.get_param("~device_id", 0)

    # 입력이 문자열로 들어올 경우를 대비해 정수로 변환 시도
    try:
        device_id = int(device_id)
    except ValueError:
        pass

    rospy.loginfo(f"카메라 장치 /dev/video{device_id} 연결 시도 중...")
    cap = cv2.VideoCapture(device_id) 

    # 카메라 열기 실패 시 로그 출력
    if not cap.isOpened():
        rospy.logerr(f"오류: /dev/video{device_id} 장치를 열 수 없습니다.")
        rospy.logerr("팁: 'ls -l /dev/video*' 명령어로 장치 번호를 확인하거나 권한을 확인하세요.")
        return

    rospy.loginfo(f"카메라(/dev/video{device_id})가 정상 작동 중입니다.")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        
        if ret:
            try:
                # OpenCV(BGR) -> ROS 메시지 변환
                img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                pub.publish(img_msg)
            except Exception as e:
                rospy.logerr(f"변환 오류: {e}")
        else:
            # 프레임을 읽지 못한 경우 (카메라 연결 끊김 등)
            rospy.logwarn_throttle(2.0, "프레임을 받아올 수 없습니다.")
        
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
