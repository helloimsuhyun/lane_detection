#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from lane_camera_pkg.drive import Driver
from motor_msgs.msg import MotorCommand
import time



class LaneSubscriber(Node):
	def __init__(self):
		super().__init__('controll_node')
		self.drv = Driver()  # 객체 인스턴스
		self.bridge = CvBridge()  # CvBridge 객체 생성 (ROS <-> OpenCV 변환)

		self.latest_frame = None # camera에서 지속적으로 들어오는 frame 넣는 변수
		self.steering_angle = None # 모터에 보낼 angle 변수
		self.moter_speed = 0 # 모터에 보낼 speed 변수 
		self.stop_line = None # stop line 확인 변수

		#화면 녹화용 변수
		self.record_out = None
		
		#fps 측정용 변수
		self.frames = 0
		self.t0 = time.perf_counter() 


		self.Moter_pub = self.create_publisher(MotorCommand, 'motor_cmd_topic', 10)
		# /camera/color/image_raw 토픽 구독
		self.sub = self.create_subscription(
			Image,  # 구독 메세지 타입
			'/camera/color/image_raw',  # 구독 토픽 이름
			self.image_callback,  # 도착할 때마다 실행할 콜백함수명
			10
		)

		# 30 fps로 이미지 처리
		self.timer = self.create_timer(1 / 30, self.line_drive)
		self.get_logger().info('✅ LaneSubscriber node started. Waiting for images...')
	
	def image_callback(self, msg):
		# ROS 이미지 메시지 → OpenCV numpy 이미지 변환
		self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
		
	def record(self,frame, video_path = "output00.avi" ):
		if self.record_out is None:
			h, w = frame.shape[:2]
			self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			self.record_out = cv2.VideoWriter(video_path, self.fourcc, 20.0, (w, h))
			if not self.record_out.isOpened():
				print("record open fail")
			else :
				print("record open")
		if len(frame.shape) == 2 or frame.shape[2] == 1:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

		r_frame = frame.copy()
		self.record_out.write(r_frame)
		return

	def record_end(self):
		if self.record_out :
			#print("recording end ! ")
			self.record_out.release()
			os.sync()
			time.sleep(1.0)
		return
		  
	def motor_control(self, speed, angle):
		if (speed != None and angle != None) :
			speed = float(speed)
			angle = float(angle)
			msg = MotorCommand()
			msg.speed = speed
			msg.angle = angle
			self.Moter_pub.publish(msg)   
		return

	def line_drive(self):
		if self.latest_frame is None:
			return
		
		frame = self.latest_frame.copy()

		img1, _, show = self.drv.pre_process(img = frame , mode = 1, show = False) # 1 : bev , 2: 색상 필터링
		_ , img2, show = self.drv.pre_process(img = frame , mode = 0, show = False)


		"""
		if isinstance(show, (list, tuple)) and len(show) >= 3:
			cv2.imshow("pre1", show[0])
			cv2.imshow("pre2", show[1])
			cv2.imshow("pre3", show[2])

		else:
			cv2.imshow("Received Image", frame)
		"""


		self.stop_line, show  = self.drv.is_stop_line(img2, show = False)
		# if (len(show) >=1) : cv2.imshow("STOP LINE DETECTION", show)
		
		self.steering_angle, show = self.drv.drive(img=img1, flag = 1 ,show = True)
		print(self.steering_angle)
		self.record(show[1],video_path = "output00.avi")


		"""
		if isinstance(show, (list, tuple)) and len(show) >= 3:
			cv2.imshow("drive1", show[0])
			cv2.imshow("drive2", show[1])
			cv2.imshow("drive3", show[2])
		else:
			cv2.imshow("Received Image", frame)
		"""


		#fps 측정
		self.frames += 1
		now = time.perf_counter()
		if now - self.t0 >= 1.0:                 # 1초마다 갱신
			fps = self.frames / (now - self.t0)
			#print(f"FPS: {fps:.2f}")
			self.t0 = now
			self.frames = 0
		
		# 모터 컨트롤 부
		if (self.stop_line == True) : self.moter_speed = 0 
		else : self.moter_speed = 20
		self.motor_control(self.moter_speed,self.steering_angle)
		cv2.waitKey(1)  # GUI 갱신
	
		
def main():
	rclpy.init()
	node = LaneSubscriber()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally : 
		print("node end")
		node.record_end()
		cv2.destroyAllWindows()
		node.destroy_node()
		if rclpy.ok() :
			rclpy.shutdown()


if __name__ == '__main__':
	main()