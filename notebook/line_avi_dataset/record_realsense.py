import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time  # 시간 측정을 위한 모듈


def record_realsense_video(output_folder='.', filename='output_video.avi', resolution=(640, 480), fps=30):
    """
    RealSense D455 카메라로 동영상을 녹화하는 함수

    Args:
        output_folder (str): 저장할 폴더 경로 (기본값: 현재 폴더)
        filename (str): 저장될 비디오 파일 이름 (기본값: 'output_video.avi')
        resolution (tuple): 비디오 해상도 (기본값: 640x480)
        fps (int): 초당 프레임 수 (기본값: 30FPS)
    """
    # 폴더 경로 존재 여부 확인
    if not os.path.exists(output_folder):
        print(f"지정한 폴더 '{output_folder}'가 존재하지 않습니다. 기본 폴더에 저장합니다.")
        output_folder = '.'  # 폴더가 없으면 기본 폴더에 저장

    # 비디오 파일 저장 경로 설정
    output_path = os.path.join(output_folder, filename)

    # 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()

    # RGB 스트림 설정 (해상도, FPS)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)

    # 스트리밍 시작
    pipeline.start(config)

    # 비디오 파일 저장 설정 (XVID 코덱, FPS, 해상도)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)

    print(f"🎥 RealSense 카메라로 영상 녹화 시작! ('{output_path}'에 저장)")

    # 시작 시간 기록
    start_time = time.time()

    frame_count = 0

    while True:
        # 프레임 캡처
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 컬러 이미지를 numpy 배열로 변환
        frame = np.asanyarray(color_frame.get_data())

        # 비디오 파일로 저장
        out.write(frame)

        # 화면에 실시간 영상 표시
        cv2.imshow("RealSense Video", frame)

        # 경과 시간 계산
        elapsed_time = time.time() - start_time
        elapsed_seconds = int(elapsed_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60

        # 경과 시간 출력
        cv2.displayOverlay("RealSense Video", f"녹화 시간: {minutes:02d}:{seconds:02d}", 1)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"✅ 영상 녹화 완료: '{output_path}'")
            break

    # 스트리밍 종료 및 리소스 정리
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()



# 저장경로, 동영상 파일 이름 저장
script_path = os.path.dirname(os.path.realpath(__file__))
print(script_path)
test_dir = os.path.join(script_path,'test') #아랫것과 택 1
train_dir = os.path.join(script_path,'train')
print(test_dir)

record_realsense_video(output_folder=test_dir, filename='04.avi', resolution=(640, 480), fps=30)

"""
aru
1. 저장할 폴더 경로 (기본값 현재 폴더) str
2. 저장할 avi 파일 네임 str
3. 해상도 기본값 (기본값: 640x480) tuple
4. 초당 프레임 수 기본값 30 

"""

#!!!!!!!!!!!!!!한번 녹화 끝날떄마다 filename 다른 이름으로 바꿔줘야함!!!!!!!!!!!!!
#!!!!!!! test train 여부 바꾸기 "output_folder = test_dir / train_dir(택1)"

