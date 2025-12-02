import cv2
import os

script_path = os.path.dirname(os.path.realpath(__file__)) #현재 파이썬 파일의 경로
test_avi_dir = os.path.join(script_path,'test') 
train_avi_dir = os.path.join(script_path,'train')
test_image_dir = os.path.join(script_path,'test/images') 
train_image_dir = os.path.join(script_path,'train/images') 

def split_video_into_images(video_path, output_folder='images', frame_interval=1, video_file_name="video"):
    """
    동영상 파일을 프레임별 이미지로 분할하는 함수

    Args:
        video_path (str): 입력할 동영상 파일 경로
        output_folder (str): 저장할 폴더 경로 (기본값: 'images')
        frame_interval (int): 몇 번째 프레임을 저장할지 간격을 설정 (기본값: 1, 모든 프레임 저장)
        video_file_name (str): 원본 동영상 파일 이름 (기본값: "video")
    """
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)

    # 동영상 파일이 열리지 않으면 오류 출력
    if not cap.isOpened():
        print("영상 파일을 열 수 없습니다.")
        return

    # 출력 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 프레임 번호 설정
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        
        # 영상 끝에 도달하면 종료
        if not ret:
            break

        frame_count += 1

        # 지정된 간격에 맞는 프레임만 저장
        if frame_count % frame_interval == 0:
            # 원본 동영상 이름을 사용하여 이미지 이름 생성
            image_name = f"{video_file_name}_{saved_count:04d}.jpg"
            image_path = os.path.join(output_folder, image_name)

            # 이미지 저장
            cv2.imwrite(image_path, frame)
            print(f"프레임 {saved_count} 저장: {image_path}")
            saved_count += 1

    # 동영상 캡처 종료
    cap.release()
    print(f"모든 프레임을 {output_folder} 폴더에 저장 완료!")


def is_train_or_test(select):

    """
    test 또는 train 폴더에서 영상 파일 경로를 선택
    Args:
        select (str): 'test' 또는 'train'을 입력받아 해당 폴더의 경로 선택
    Returns:
        tuple(str): 선택된 폴더의 비디오 파일 경로(분할할 파일의 경로), 
    """

    if select == 'test':
        return os.path.join(test_avi_dir, video_file_name), test_image_dir
    elif select == 'train':
        return os.path.join(train_avi_dir, video_file_name), train_image_dir
    else:
        raise ValueError("잘못된 선택입니다. 'test' 또는 'train'을 입력하세요.")


#########################################아래가 인풋 부분 

# 1. 분할하고 싶은 동영상 파일의 파일명을 기입
video_file_name = '03.avi'
# 2. 분할할 영상의 test 혹은 train여부 선택
video_file_path, image_file_path = is_train_or_test('train') 

# 3. 분할할 프레임 간격수 - frame_interval 만 수정해주면 된다.
split_video_into_images(
    video_path=video_file_path,
    output_folder=image_file_path,
    frame_interval=20,
    video_file_name=os.path.splitext(video_file_name)[0]  
)

