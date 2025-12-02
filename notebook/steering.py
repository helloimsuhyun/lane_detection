import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def pre_process_1(img):
    """
    튜닝 파라미터 -> 색공간 노랑, 흰색 필터링 배열
    """
    #1. resize
    img_conv = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)  # 크기 조정

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

def pre_process_2(img, roi = [[70, 350],[570, 350],[640, 480],[0, 480]]):
    """
    bird eye view 변환
    """
    # bird's eye view를 적용할 roi를 넘파이 배열로 지정
    src = np.array(roi, dtype=np.float32) #좌상단 우상단 우하단 좌하단
    width, height = 500, 420
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]]) #좌상단 우상단 우하단 좌하단
    
    # 투시 변환 행렬 
    M = cv2.getPerspectiveTransform(src, dst)
    # 원근 변환 적용
    img_conv = cv2.warpPerspective(img, M, (width, height))
    img_conv = cv2.Canny(image=img_conv, threshold1=50, threshold2=150)

    #cv2.imshow("Lane Preprocessed", img_conv)

    return img_conv

def debug_draw_roi(img, 
                   roi_list=[[70, 350], [570, 350], [640, 480], [0, 480]],
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



def find_base(img, basefindgrid_N = 20):

    """
    1. sliding window base 위치 찾기 
        - 받아온 이미지를 가로로 나누어 만든 N개의직사각형 중, 비트합을 하였을 때, midpoint를 기준으로 좌측과 우측의 최댓값을 base로 선정
    """
    H,W = img.shape
    
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

    left_max_idx = np.argmax(findbase_score[:basefindgrid_midpoint])
    right_max_idx = np.argmax(findbase_score[basefindgrid_midpoint:])  
    right_max_idx = basefindgrid_midpoint + right_max_idx

    left_max_idx = int(left_max_idx)
    right_max_idx = int(right_max_idx)

    x1,x2 = findbase_edge[left_max_idx]
    leftx_base = (x1+x2)//2

    x1,x2 = findbase_edge[right_max_idx]
    rightx_base = (x1+x2)//2

    return leftx_base,rightx_base

def find_line_point(img,leftx_base,rightx_base, nwindows = 20, h_margin = 60, l_margin = 80, minpix = 40):
    """
    nwindows
    h_margin
    l_margin
    minpix

    """
    # 슬라이딩 윈도우 설정
    nwindows = 15
    window_height = img.shape[0] // nwindows
    h_margin = 60
    l_margin = 100
    minpix = 50
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


def find_line_fit(left_box_clean,right_box_clean):
    L_coff = None
    R_coff = None

    if (len(left_box_clean) >= 8):
        left_box_clean = np.array(left_box_clean,dtype=float)
        xL, yL = left_box_clean[:,0], left_box_clean[:,1]
        L_coff = tuple(np.polyfit(yL, xL, 2))
    else :
        print("no LEFT line detected")
        
        
    if (len(right_box_clean) >= 8):
        right_box_clean = np.array(right_box_clean,dtype=float)
        xR, yR = right_box_clean[:,0], right_box_clean[:,1]
        R_coff = tuple(np.polyfit(yR, xR, 2))
    else :
        print("no RIGHT line detected")

    if L_coff is not None:
        print("L_coff:", L_coff)
    if R_coff is not None:
        print("R_coff:", R_coff)

    return L_coff, R_coff

def debug_show_line_fit_points(img, L_coff=None, R_coff=None,
                               left_color=(255, 0, 0), right_color=(0, 0, 255),
                               radius=3, step=10, copy=True):
    """
    img: GRAY or BGR 이미지
    L_coff, R_coff: np.polyfit 결과 (a, b, c)
    left_color, right_color: 점 색상
    radius: 점 크기
    step: y축 점 간격 (작을수록 곡선이 촘촘함)
    copy: 원본 보존 여부
    """
    vis = img.copy() if copy else img
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    H, W = vis.shape[:2]

    # 좌측 곡선 점 찍기
    if L_coff is not None:
        a, b, c = L_coff
        for y in range(0, H, step):
            x = int(a * y**2 + b * y + c)
            if 0 <= x < W:
                cv2.circle(vis, (x, y), radius, left_color, -1, lineType=cv2.LINE_AA)

    # 우측 곡선 점 찍기
    if R_coff is not None:
        a, b, c = R_coff
        for y in range(0, H, step):
            x = int(a * y**2 + b * y + c)
            if 0 <= x < W:
                cv2.circle(vis, (x, y), radius, right_color, -1, lineType=cv2.LINE_AA)

    return vis


def find_angle(img, L_coff, R_coff, lane_width = 250):
    H, W = img.shape
    
    if L_coff == None and R_coff == None :
        return None, None, None
    elif L_coff != None and R_coff == None :
        a, b, c = L_coff
        R_coff = (a, b, c + lane_width)
    elif R_coff != None and L_coff == None :
        a, b, c = R_coff
        L_coff = (a, b, c - lane_width)
    
    aL , bL , cL = L_coff
    aR , bR, cR = R_coff
    
    lookahead_ys = [int(H * 0.2), int(H * 0.4), int(H * 0.6), int(H * 0.8)]

    angles = []  # 각 지점에서 조향각 저장

    for y in lookahead_ys:
        # 차선 중심 계산
        left_x = aL * y**2 + bL * y + cL
        right_x = aR * y**2 + bR * y + cR
        lane_center = (left_x + right_x) / 2

        # 차량 중심과 차선 중심의 차이
        dx = lane_center - (W / 2)
        dy = H - y

        # 조향각 계산
        angle = np.arctan2(dx, dy)
        angles.append(angle)

    # 3. 평균 조향각 계산
    if len(angles) > 0:
        steering_angle_rad = sum(angles) / len(angles)
        steering_angle_deg = np.degrees(steering_angle_rad)
        print("측정 조향각 (deg):", steering_angle_deg)
    else:
        print("조향각을 계산할 수 없습니다.")
    
    return steering_angle_deg, L_coff, R_coff



def draw_steering_angle(img, steering_angle_deg, length=150, color=(0, 255, 0), thickness=4):
        """
        img: BEV 또는 원본 이미지 (BGR)
        steering_angle_deg: 조향각 (deg)
        length: 화살표 길이 (픽셀)
        color: 화살표 색상 (BGR)
        thickness: 선 두께
        """

        vis = img.copy() 
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


import time

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

    def step(self, meas_deg, dt):
        self._set_F_Q(dt)
        x_pre = self.kf.predict()  # statePre
        if meas_deg is not None:
            z = np.array([[float(meas_deg)]], np.float32)
            self.kf.correct(z)
            return float(self.kf.statePost[0,0])
        else:
            return float(x_pre[0,0])


angle_kf = KalmanAngleSimple(init_theta=0.0, Q_acc=0.8, R_meas=12.0)

"""
last_angle = 0.0      # 마지막(필터링된) 각도
SMOOTH = 0.8          # 0.7~0.95 추천. 클수록 더 부드럽고 느림
"""
prev_time = None

if __name__ == "__main__" : 

    """

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

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir,'data/line_avi_dataset/data/videos/train','02.avi')

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 끝까지 읽으면 종료

        # ---- 기존 이미지 파이프라인 그대로 ----
        mask = pre_process_1(frame)
        bev  = pre_process_2(mask)
        show4 = debug_draw_roi(mask)

        left_base, right_base = find_base(bev)
        L_pts, R_pts = find_line_point(bev, left_base, right_base)
        show3 = debug_show_line_point(bev, L_pts, R_pts)
        L_coff, R_coff = find_line_fit(L_pts, R_pts)

        # 칼만 필터 보정 적용
        #if L_coff is not None:
        #    L_coff = kalman_update(kf_left, L_coff)
        #if R_coff is not None:
        #    R_coff = kalman_update(kf_right, R_coff)


        meas , lc, rc = find_angle(bev, L_coff, R_coff)  # 조향각 관측값
        show1 = debug_show_line_fit_points(bev, lc, rc)

        now = time.time()
        if prev_time is None:
            dt = 1/20.0            # 초기값(대략 FPS가 20이라 가정)
        else:
            dt = max(1e-3, min(0.2, now - prev_time))  # 0.001~0.2s 범위로 제한
        prev_time = now
                

        # --- 칼만 필터로 각도 추정 ---
        angle_deg = angle_kf.step(meas_deg=meas, dt=dt)
        print(angle_deg)

        """
        if meas ==  None:
            angle_deg = last_angle          # 관측 없으면 '유지'(0으로 리셋 금지)
        else:
            # 지수이동평균
            angle_deg = SMOOTH*last_angle + (1-SMOOTH)*meas

        last_angle = angle_deg  # 상태 업데이트
        """

        # angle_deg를 이용해 직선 시각화
        show2 = draw_steering_angle(bev, angle_deg, length=200, color=(0, 255, 255))

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

