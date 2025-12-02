#!/usr/bin/env python3
import cv2
import time

class SimpleCamera:
    def __init__(self, cam_id=6, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #self.cap.set(cv2.CAP_PROP_FPS, 30)

                # --- 자동 화이트밸런스 끄기 ---
        ok1 = self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        # --- 화이트밸런스 온도 고정 (켈빈) ---
        ok2 = self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Camera {cam_id} not accessible")
        
        print(f"[OpenCV] AUTO_WB off: {ok1}, WB_TEMPERATURE set({4600}K): {ok2}")
        if not (ok1 and ok2):
            print("⚠️ OpenCV로 WB 설정이 적용되지 않을 수 있습니다. (백엔드 미지원 가능)")

    def publish_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

"""
lusub 현재 usb 연결 포트 확인

v4l2-ctl --list-devices

for i in {2..7}; do
  echo "===== /dev/video$i ====="
  v4l2-ctl --device=/dev/video$i --list-formats-ext | grep -E "MJPG|YUYV|RGB"
done

카메라 채널 확인하기
"""

if __name__ == "__main__":
    cam = SimpleCamera()
    prev_time = time.time()
    fps = 0.0

    while True:
        frame = cam.publish_frame()
        if frame is None:
            print("⚠️ Frame read failed")
            break

        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        # --- FPS 계산 ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        # --- 화면에 FPS 표시 ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Camera Stream", frame)

        # ESC 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.cap.release()
    cv2.destroyAllWindows()
