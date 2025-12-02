import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import math

def pre_process_1(img, resize = (640, 480)):
    """
    튜닝 파라미터 -> 색공간 노랑, 흰색 필터링 배열
    """
    #1. resize
    img_conv = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)  # 크기 조정

    #2. 색공간 필터링 - hls, hsv
    #색공간 변경 HLS로 white 검출, HSV로 노란색 차선 검출
    hls = cv2.cvtColor(img_conv, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img_conv, cv2.COLOR_BGR2HSV)

    # 흰색 차선 (HLS) - 높은 L = 흰색
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 255])
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    # 노란색 차선 (HSV) - 10~40의 H 노란색
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    img_conv = cv2.bitwise_or(mask_white,mask_yellow)

    #3. 가우시안블러 , 모폴로지연산으로 noize 제거
    img_conv = cv2.GaussianBlur(img_conv, (5, 5), 0)  # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_conv = cv2.morphologyEx(img_conv, cv2.MORPH_OPEN, kernel)
    img_conv = cv2.morphologyEx(img_conv, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow("Lane Preprocessed", img_conv)

    return img_conv

def pre_process_2(img, roi = [[70, 350], [570, 350], [640, 480], [0, 480]], bev_size = (500, 420)):
    """
    bird eye view 변환
    """
    # bird's eye view를 적용할 roi를 넘파이 배열로 지정
    src = np.array(roi, dtype=np.float32) #좌상단 우상단 우하단 좌하단
    width, height = bev_size
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]]) #좌상단 우상단 우하단 좌하단
    
    # 투시 변환 행렬 
    M = cv2.getPerspectiveTransform(src, dst)
    # 원근 변환 적용
    img_conv = cv2.warpPerspective(img, M, (width, height))
    img_conv = cv2.Canny(image=img_conv, threshold1=50, threshold2=150)

    #cv2.imshow("Lane Preprocessed", img_conv)

    return img_conv

def debug_draw_roi(img, 
                   roi_list,
                   color=(0, 255, 0),
                   thickness=2,
                   show=True):
    """
    ROI를 시각화해서 디버그용으로 그려주는 함수

    Parameters
    ----------
    img : np.ndarray
        원본 이미지 (GRAY or BGR)
    roi_list : list
        ROI 좌표 리스트 [[x1, y1], [x2, y2], ...]
    color : tuple
        ROI 라인 색상 (B, G, R)
    thickness : int
        ROI 라인 두께
    show : bool
        True면 matplotlib으로 즉시 시각화

    Returns
    -------
    vis : np.ndarray
        ROI가 그려진 BGR 이미지
    """
    # 1. 복사 & 컬러 변환
    img_copy = img.copy()
    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    # 2. ROI 라인 그리기
    pts = np.array(roi_list, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return img_copy


def find_base(img, past_L_base_idx, past_R_base_idx, basefindgrid_N = 20, thresh_score = 40000, find_base_width_idx = 3, mode = 0, show = False):

    """
    sliding window base 위치 찾기 
    img : 이미지
    past_L_base_idx : 이전의 L 라인 베이스 (None / int)
    past_R_base_idx : 이전의 R 라인 베이스 (None / int)
    basefindgrid_N : 몇개의 그리드로 나눠서 찾을지 
    thresh_score : 라인이 존재한다는 그리드 탐색 임개값
    find_base_width_idx : 이전 라인에서 주변 몇개 그리드에서 탐색할지
    mode = 0 : 0이면 이전 값을 참조하여 라인을 찾음 , 1이면 그냥 중간부터 해서 base 탐색
    show = Flase : True이면 plt로 socore그래프, 찾은 베이스를 프린팅

    return  leftx_base_idx rightx_base_idx 이번에 찾은 base의 인덱스

    ** 각 호출간의 basefindgrid_N이 동일해야함
    """
    H,W = img.shape

    # 파라미터 가드
    if basefindgrid_N < 2 : 
        basefindgrid_N = 2
    if find_base_width_idx < 1 : 
        find_base_width_idx = 1
    if(mode != 0 and mode != 1) :
        mode = 1
    
    # 이미지를 N개의 그리드로 나눠 각 그리드 안의 픽셀값을 findbase_score에 저장, 각 그리드의 좌표값을 findbase_edge에 저장
    basefindgrid_margin = W // basefindgrid_N
    basefindgrid_midpoint = basefindgrid_N // 2

    findbase_score = []
    findbase_edge = []

    for i in range(basefindgrid_N) :
        x1 = i * basefindgrid_margin
        x2 = W if i == basefindgrid_N -1 else (i+1)*basefindgrid_margin # 마지막 grid일 경우 대비 보정항
        
        temp = np.sum(img[:, x1:x2])
        findbase_score.append(temp)
        findbase_edge.append((x1,x2))

    findbase_score = np.array(findbase_score)

    left_th_idx = basefindgrid_midpoint - find_base_width_idx
    right_th_idx = basefindgrid_midpoint + find_base_width_idx

    # 2. 이전 각 좌/우 라인 베이스 기준 폭 2*find_base_width_idx만큼을 검사 -> 최고인 것을 베이스로 선정
    if (mode == 0 and past_L_base_idx != None) :
        basefind_idx_L_1 = past_L_base_idx - find_base_width_idx
        basefind_idx_L_2 = past_L_base_idx + find_base_width_idx
        if (basefind_idx_L_1 <= 0) :  basefind_idx_L_1 = 0
        if (basefind_idx_L_2 >= basefindgrid_midpoint) :  basefind_idx_L_2 = left_th_idx

        left_max_idx = np.argmax(findbase_score[basefind_idx_L_1: basefind_idx_L_2]) #   basefind_idx_L_1 ~  basefind_idx_L_2 -1 까지 검사
        left_max_idx = basefind_idx_L_1 + left_max_idx
        left_max_idx = int(left_max_idx)


    if (mode == 0 and past_R_base_idx != None) :

        basefind_idx_R_1 = past_R_base_idx - find_base_width_idx
        basefind_idx_R_2 = past_R_base_idx + find_base_width_idx
        if (basefind_idx_R_1 <= basefindgrid_midpoint) :  basefind_idx_R_1 = right_th_idx
        if (basefind_idx_R_2 >= len(findbase_score)) :  basefind_idx_R_2 = len(findbase_score)

        right_max_idx = np.argmax(findbase_score[basefind_idx_R_1 :basefind_idx_R_2]) # basefind_idx_R_1 ~  basefind_idx_R_2 -1 까지 검사
        right_max_idx = basefind_idx_R_1 + right_max_idx
        right_max_idx = int(right_max_idx)


    # 3. 만약 이전 라인 주변에서 찾은 base의 score가 일정 수치 이하일 경우, 중앙점을 기준으로 한 검사를 다시 실행 -> 해당 지점을 base로 수정

    if (mode == 1 or past_L_base_idx == None or findbase_score[left_max_idx] <= thresh_score ) :
        left_max_idx = np.argmax(findbase_score[:left_th_idx]) # 0 ~ basefindgrid_midpoint-1 까지 검사
        left_max_idx = int(left_max_idx)

    if (mode == 1 or past_R_base_idx == None or findbase_score[right_max_idx] <= thresh_score) :
        right_max_idx = np.argmax(findbase_score[right_th_idx:]) # basefindgrid_midpoint ~ 끝까지 검사
        right_max_idx = right_th_idx + right_max_idx
        right_max_idx = int(right_max_idx)


    x1,x2 = findbase_edge[left_max_idx]
    leftx_base = (x1+x2)//2

    x1,x2 = findbase_edge[right_max_idx]
    rightx_base = (x1+x2)//2

    # 4. 이번에 찾은 인덱스를 리턴, 각선의 base x 좌표를 리턴
    leftx_base_idx  = left_max_idx
    rightx_base_idx = right_max_idx

    
    if (show == True) :
        print(f"""
            total idx : {basefindgrid_N}
            finded Left base idx  : {leftx_base_idx}
            finded Right base idx : {rightx_base_idx}
            
            finded Left base socore  : {findbase_score[leftx_base_idx]}
            finded Right base socore  : {findbase_score[rightx_base_idx]}

            past left base : {past_L_base_idx}
            past left base : {past_R_base_idx}
            """)
        #plt.plot(findbase_score)
        #plt.grid(True)
        #plt.show()

    return leftx_base,rightx_base, leftx_base_idx,rightx_base_idx


def find_line_point(img,leftx_base,rightx_base, nwindows = 20, h_margin = 60, l_margin = 80, minpix = 20):
    """
    nwindows
    h_margin
    l_margin
    minpix

    """
    # 슬라이딩 윈도우 설정

    window_height = img.shape[0] // nwindows
    maxpix = 200

    leftx_current = leftx_base
    rightx_current = rightx_base

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    left_box_centor = []
    right_box_centor = []

 
    """
    1. 처음 n개의 윈도우 크기와 위의 window 크기를 다르게 적용
    2. 각 window에서 감지된 차선 픽셀들의 무게중심 계산
        감지된 픽셀이 임계치 이하인 경우 차선이 없음으로 인식 -> None 처리
    3. 각 무게중심을 이어서 2차함수로 피팅
    """

    for window in range(nwindows):
        
        if (window <= 2 and window >= 0): 
            margin = l_margin
        else : 
            margin = h_margin
            
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 윈도우 내 픽셀 인덱스 추출
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 윈도우의 무게중심 위치를 centor 변수에 저장, 리스트에 넣음

        if len(good_left_inds) > minpix :
            left_x_centor = int(np.mean(nonzerox[good_left_inds]))
        else :
            #print(len(good_left_inds))
            left_x_centor = None
            
        if len(good_right_inds) > minpix : 
            right_x_centor = int(np.mean(nonzerox[good_right_inds]))

        else : 
            right_x_centor = None

        y_centor = (win_y_low+win_y_high)//2
        left_box_centor.append((left_x_centor,y_centor))
        right_box_centor.append((right_x_centor,y_centor))
        
        if len(good_left_inds) > minpix :
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: 
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_box_clean = [p for p in left_box_centor if p[0] is not None]
    right_box_clean = [p for p in right_box_centor if p[0] is not None]
 
    return left_box_clean, right_box_clean


def debug_show_line_point(img, left_box_clean, right_box_clean,
                          left_color=(255, 0, 0), right_color=(0, 0, 255),
                          radius=3, thickness=-1, draw_polyline=True, copy=True):
    """
    img: GRAY 또는 BGR 이미지
    left_box_clean/right_box_clean: [(x, y), ...]  # 이미 None 제거된 포인트
    left_color/right_color: BGR 색
    radius: 점 반지름, thickness=-1이면 채운 점
    draw_polyline: 점들을 순서대로 선으로 이어서 확인하고 싶을 때 True
    copy: 원본 훼손 방지용. True면 복사본에 그림

    return: 시각화된 BGR 이미지
    """
    # BGR 준비
    vis = img.copy() if copy else img
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 좌/우 점 찍기
    for (x, y) in left_box_clean:
        cv2.circle(vis, (int(x), int(y)), radius, left_color, thickness, lineType=cv2.LINE_AA)
    for (x, y) in right_box_clean:
        cv2.circle(vis, (int(x), int(y)), radius, right_color, thickness, lineType=cv2.LINE_AA)

    # 간단한 카운트 표시
    text = f"L:{len(left_box_clean)}  R:{len(right_box_clean)}"
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

    return vis


def find_line_fit(left_box_clean,right_box_clean, line_width= 200, flag = 0, img_mid = 250, bias = 50):
    x_current = None
    x_location = None
    current_line = "MID"

    if len(left_box_clean) == 0 and len(right_box_clean) == 0 :
        return x_location, x_current , current_line
    if (flag != 0 and flag != 1 and flag != 2):
        flag = 0
    
    if (flag == 0): # 일반 주행 모드
        if (len(left_box_clean) > 15 and len(right_box_clean) > 15) :
            x1 = [x for x,_ in left_box_clean] 
            x2 = [x for x,_ in right_box_clean]
            x = x1+x2 
            x_current = float(np.mean(x))
            x_location = x_current
            current_line = "LEFT/RIGHT"

        elif (len(left_box_clean) >  len(right_box_clean)):
            x = [x for x,_ in left_box_clean]
            x_current = float(np.mean(x))
            x_location = x_current + float(line_width)
            current_line = "LEFT"

        elif(len(left_box_clean) <  len(right_box_clean)):
            x = [x for x,_ in right_box_clean]
            x_current = float(np.mean(x))
            x_location = x_current - float(line_width)
            current_line = "RIGHT"

        else :
            current_line = "NO LANE"

    elif (flag == 1) : # 왼쪽 차선만 따라가는 모드
        if (len(left_box_clean) > 10):
            x = [x for x,_ in left_box_clean]
            x_current = float(np.mean(x))
            x_location = x_current + float(line_width)
            current_line = "LEFT"
        else : 
            x_location = float(img_mid) - float(line_width) + float(bias)
            current_line = "NO LANE / / SEARCH LEFT"
    
    elif (flag == 2) : # 우측 차선만 따라가는 모드
        if (len(right_box_clean) > 10):
            x = [x for x,_ in right_box_clean]
            x_current = float(np.mean(x))
            x_location = x_current - float(line_width)
            current_line = "RIGHT"
        else : 
            x_location = float(img_mid) + float(line_width) - float(bias)
            current_line = "NO LANE / SEARCH RIGHT"

    return x_location, x_current , current_line
       

def debug_show_line_fit_points(img, x_location, x_current, current_line, y=340):

    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    H, W = vis.shape[:2]

    center_x = W // 2
    cv2.line(vis, (center_x, 0), (center_x, H), (255, 255, 0), 1)  # BGR: 하늘색

    # x_location 점 (초록색)
    if x_location is not None:
        cv2.circle(vis, (int(x_location), y), 5, (0, 255, 0), -1)
        cv2.putText(vis, f"x_loc: {int(x_location)}", (int(x_location) - 40, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # x_current 점 (빨간색)
    if x_current is not None:
        cv2.circle(vis, (int(x_current), y), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"x_cur: {int(x_current)}", (int(x_current) - 40, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 현재 추종 차선 텍스트
    cv2.putText(vis, f"LINE: {current_line}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def find_angle(img, x_location, last_angle = 0.0, Kp = 0.003, max_deg=20.0):
    H, W = img.shape
    img_center_x = W /2

    if x_location is None:
        return last_angle  # 인식 실패 시 이전 각 유지(간단한 fail-safe)
    
    e = float(x_location) - float(img_center_x)
    angle_deg = Kp * e * (180 / math.pi)  # rad → deg 변환

    # 각도 제한
    angle_deg = max(-max_deg, min(max_deg, angle_deg))

    return angle_deg


def draw_steering_angle(img, steering_angle_deg, length=150, color=(0, 255, 0), thickness=4):
        """
        img: BEV 또는 원본 이미지 (BGR)
        steering_angle_deg: 조향각 (deg)
        length: 화살표 길이 (픽셀)
        color: 화살표 색상 (BGR)
        thickness: 선 두께
        """

        vis = img.copy() 
        if (steering_angle_deg != None) : 
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            H, W = vis.shape[:2]


            # 차량 기준점 (아래쪽 중앙)
            center = (W // 2, H)

            # 각도를 라디안으로 변환
            angle_rad = np.radians(steering_angle_deg)

            # 조향 벡터의 끝점 계산
            end_x = int(center[0] + length * np.sin(angle_rad))
            end_y = int(center[1] - length * np.cos(angle_rad))

            # 화살표 그리기
            cv2.arrowedLine(vis, center, (end_x, end_y), color, thickness, tipLength=0.2)

            # 조향각 텍스트 표시
            cv2.putText(vis, f"Steering: {steering_angle_deg:.2f} deg", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return vis



### 칼만


class KalmanAngleSimple:
    """
    x = [theta_deg, theta_rate_deg_s]^T
    z = [theta_deg]
    """
    def __init__(self, init_theta=0.0, Q_acc=1.0, R_meas=9.0):
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix  = np.eye(2, dtype=np.float32)     # F는 매 step에서 dt 반영
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)  # H
        self.Q_acc = float(Q_acc)                                    # 가속도 잡음 파워
        self.kf.measurementNoiseCov = np.array([[R_meas]], np.float32)
        self.kf.statePost    = np.array([[init_theta],[0.0]], np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 10.0

    def _set_F_Q(self, dt):
        dt = float(max(1e-3, dt))
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        self.kf.transitionMatrix = F
        # CV의 표준 Q(dt)
        dt2, dt3 = dt*dt, dt*dt*dt
        q11 = self.Q_acc * dt3/3.0
        q12 = self.Q_acc * dt2/2.0
        q22 = self.Q_acc * dt
        Q = np.array([[q11, q12],
                      [q12, q22]], dtype=np.float32)
        self.kf.processNoiseCov = Q

    def step(self, meas, dt):
        self._set_F_Q(dt)
        x_pre = self.kf.predict()  # statePre
        if meas is not None:
            z = np.array([[float(meas)]], np.float32)
            self.kf.correct(z)
            return float(self.kf.statePost[0,0])
        else:
            return float(x_pre[0,0])




######################################전역변수
g_past_L_base_idx = None
g_past_R_base_idx = None
angle_kf = KalmanAngleSimple(init_theta=0.0, Q_acc=0.8, R_meas=12.0)
g_prev_time = None
g_last_angle = 0
######################################전역변수


if __name__ == "__main__" : 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir,'data/line_avi_dataset/data/videos/train','01.avi')

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 끝까지 읽으면 종료
        
        """
        주요 구동함수 
        pre_process_1
        pre_process_2
        (debug_draw_roi)
        find_base
        find_line_point
        (debug_show_line_point)
        find_line_fit
        (debug_show_line_fit_points)
        find_angle
        (draw_steering_angle)
        
        전역변수
        past_L_base_idx
        past_R_base_idx
            -- 베이스 찾기용
        prev_time
        angle_deg 
            -- 베이스 제어, 칼만 필터링용

        """

        """원래 사용 resize = (640, 480)
        roi = [[130, 350], [540, 350], [570, 430], [100, 430]]
        bev_size = (500, 420)
        """
        """
        인풋 320, 240으로 받아오면 
        mask = pre_process_1(frame,
                             resize = (320, 240)
                             )
        roi = [[65,175], [270,175], [285,215], [50,215]]
        bev_size = (500, 420)

        """
        mask = pre_process_1(frame,
                             resize = (320, 240)
                             )
        roi = [[65,175], [270,175], [285,215], [50,215]]
        bev_size = (500, 420)

        bev  = pre_process_2(mask,
                            roi = roi,
                            bev_size = bev_size
                            )
        show4 = debug_draw_roi(mask,roi_list = roi) 

        left_base, right_base , left_base_idx , right_base_idx = find_base(bev, 
                                        past_L_base_idx = g_past_L_base_idx,
                                        past_R_base_idx = g_past_R_base_idx,
                                        basefindgrid_N = 20, 
                                        thresh_score = 40000, 
                                        find_base_width_idx = 3, 
                                        mode = 0,
                                        show = False)
        
        g_past_L_base_idx = left_base_idx
        g_past_R_base_idx = right_base_idx

        L_pts, R_pts = find_line_point(bev, left_base, right_base)
        show3 = debug_show_line_point(bev, L_pts, R_pts)
            
        x_location, x_current , current_line = find_line_fit(L_pts,R_pts, line_width= 200, flag = 0)
        show1 = debug_show_line_fit_points(bev,x_location,x_current,current_line)
        
        now = time.time()
        if g_prev_time is None:
            dt = 1/20.0            # 초기값(대략 FPS가 20이라 가정)
        else:
            dt = max(1e-3, min(0.2, now - g_prev_time))  # 0.001~0.2s 범위로 제한ㅂ
        g_prev_time = now
                
        # --- 칼만 필터로 각도 추정 ---
        xxx = angle_kf.step(meas=x_location, dt=dt)
        angle = find_angle(bev, xxx , g_last_angle)  # 조향각 관측값
        g_last_angle = angle

        # angle_deg를 이용해 직선 시각화
        show2 = draw_steering_angle(bev, angle, length=200, color=(0, 255, 255))

        cv2.imshow("angle", show2)
        cv2.imshow("line fitting", show1)
        cv2.imshow("sliding window point", show3)
        cv2.imshow("roi", show4)

        # ---- 화면에 출력 ----
        

        # ESC나 q 누르면 종료
        if cv2.waitKey(50) & 0xFF in [27, ord('q')]:
            break
    
    print("end")
    cap.release()
    cv2.destroyAllWindows()


    """
    한개 이미지로 테스트
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir,'data/line_avi_dataset/data/images/train/extracted_20f','01_0012.jpg')

    img = cv2.imread(image_path)
    mask = pre_process_1(img)  # 전처리1 색상필터링/가우시안블러
    bev  = pre_process_2(mask)     # BEV + Canny
    left_base, right_base = find_base(bev)


    L_pts, R_pts = find_line_point(bev, left_base, right_base,
                                   nwindows=20, h_margin=60, l_margin=80, minpix=40)
    show = debug_show_line_point(bev,L_pts,R_pts)
    
    L_coff, R_coff = find_line_fit(L_pts, R_pts)

    _ = find_angle(bev, L_coff, R_coff)


    cv2.imshow("BEV", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
