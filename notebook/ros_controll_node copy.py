#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32

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
		self.stop_line = False # stop line 확인 변수

		#제어변수
		self.state = "NORMAL"
		self.stop_cnt = 0
		self.stamp_time = 0.0

		#화면 녹화용
		self.record_out = None
		self.recordwhat = 3 # 0 녹화하지 않음, 1 원본영상 녹화 , 2 색상필터링 + bev roi 녹화 3 정지천 감지 녹화 4 차선추종 녹화
		self.record_show = None
		
		#fps 측정용 변수
		self.frames = 0
		self.t0 = time.perf_counter()

		#lider 장애물 확인 변수
		self.obstacle_flag = False
		self.dist = [999.0, 999.0, 999.0]

		self.Moter_pub = self.create_publisher(MotorCommand, 'motor_cmd_topic', 10)
		# /camera/color/image_raw 토픽 구독
		self.sub = self.create_subscription(
			Image,  # 구독 메세지 타입
			'/camera/color/image_raw',  # 구독 토픽 이름
			self.image_callback,  # 도착할 때마다 실행할 콜백함수명
			10
		)

		self.lider_dist_sub = self.create_subscription(
			Float32MultiArray,  # 구독 메세지 타입
			'/obstacle_distances',  # 구독 토픽 이름
			self.lider_dist_callback,  # 도착할 때마다 실행할 콜백함수명
			10
		)

		self.lidar_angle_sub = self.create_subscription(
			Float32, 
			'/steering_cmd_deg', 
			self.lider_angle_callback,
			10
		)

		# 30 fps로 이미지 처리
		self.timer = self.create_timer(1 / 30, self.line_drive)
		self.get_logger().info('✅ LaneSubscriber node started. Waiting for images...')
	
	def image_callback(self, msg):
		# ROS 이미지 메시지 → OpenCV numpy 이미지 변환
		self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
	
	def lider_dist_callback(self,msg):
		self.dist = msg.data
	
	def lider_angle_callback(self, msg):
		self.lidar_angle = msg.data
		
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
		now = time.perf_counter()
		
		self.stop_detect()
		
		if False :
			## 4초간 정지 후, stop line 안보일때까지 전진 후 다시 라인트레이싱 코드
			if self.state == "NORMAL" : 
				if (self.stop_line == True):
					self.state = "STOP"
					self.stamp_time = now
				else :
					self.follow_line()

			elif self.state == "STOP" :
				if (now - self.stamp_time <= 4.0) :
					self.stop()
				else :
					self.state = "GO_TO_OUTOFSTOP"

			elif self.state == "GO_TO_OUTOFSTOP": 
				if (self.stop_line == True) : 
					self.go()
				else :
					self.stop_cnt += 1
					self.state = "NORMAL"
				
			#fps 측정
			self.frames += 1
			now = time.perf_counter()
			if now - self.t0 >= 1.0: # 1초마다 갱신
				fps = self.frames / (now - self.t0)
				print(f"FPS: {fps:.2f}")
				self.t0 = now
				self.frames = 0


			if self.record_show is not None :
				self.record(self.record_show)
			
			print(self.state , self.stop_cnt , self.dist[1])

		else :
			self.lidar_drive()
			
	
	def lidar_drive(self) :
		self.steering_angle = self.lidar_angle
		print(self.steering_angle)
		self.moter_speed = 20
		self.motor_control(self.moter_speed,self.steering_angle)

	def stop_detect(self) :
		debug = False
		debug_ = False

		if self.recordwhat >= 1 and self.recordwhat <= 2 : 
			debug = True
		elif self.recordwhat == 3 :
			debug_ = True

		frame = self.latest_frame.copy()
		_ , frame, show = self.drv.pre_process(img = frame , mode = 0, show = debug)
		self.stop_line, show_  = self.drv.is_stop_line(frame, show = debug_)
		
		if self.recordwhat == 1 : self.record_show = show[0] #원본영상 녹화
		elif self.recordwhat == 2 : self.record_show = show[1] #필터링 및 roi 녹화
		elif self.recordwhat == 3 : self.record_show = show_ #정지선 감지 녹화

	def follow_line(self , line_width = 200) :
		debug = False
		if self.recordwhat == 4 : debug = True
		frame = self.latest_frame.copy()
		frame, _, show = self.drv.pre_process(img = frame , mode = 1, show = False) # 1 : bev , 2: 색상 필터링
		self.steering_angle, show = self.drv.drive(img=frame, flag = 0 ,line_width = line_width , show = debug)

		if self.recordwhat == 4 : self.record_show = show[2] # x 추정 녹화

		self.moter_speed = 20
		self.motor_control(self.moter_speed,self.steering_angle)

	def follow_left_line(self, line_width = 200 , bias = 50) :
		debug = False

		if self.recordwhat == 4 : debug = True
		frame = self.latest_frame.copy()
		frame, _, show = self.drv.pre_process(img = frame , mode = 1, show = False) # 1 : bev , 2: 색상 필터링
		self.steering_angle, show = self.drv.drive(img=frame, flag = 1 ,line_width = line_width , bias = bias, show = True)

		if self.recordwhat == 4 : self.record_show = show[2] # x 추정 녹화
		self.moter_speed = 20
		self.motor_control(self.moter_speed,self.steering_angle)

	def follow_right_line(self, line_width = 200 , bias = 50) :
		debug = False

		if self.recordwhat == 4 : debug = True
		frame = self.latest_frame.copy()
		frame, _, show = self.drv.pre_process(img = frame , mode = 1, show = False) # 1 : bev , 2: 색상 필터링
		self.steering_angle, show = self.drv.drive(img=frame, flag = 2 ,line_width = line_width , bias = bias, show = True)

		if self.recordwhat == 4 : self.record_show = show[2] # x 추정 녹화
		self.moter_speed = 20
		self.motor_control(self.moter_speed,self.steering_angle)

	
	def go(self) :
		self.moter_speed = 20 
		self.steering_angle = 0
		self.motor_control(self.moter_speed,self.steering_angle)
		

	def stop(self) : 

		self.steering_angle = 0
		self.moter_speed = 0 
		self.motor_control(self.moter_speed,self.steering_angle)


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