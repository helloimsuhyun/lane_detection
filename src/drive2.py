import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import math

def pre_process_1(img, lower_white, upper_white, lower_yellow , upper_yellow , resize = (640, 480), mode = 0):
    """
    mode 0 노랑 흰 1 노랑 2 흰
    튜닝 파라미터 -> 색공간 노랑, 흰색 필터링 배열
    """
    #1. resize
    img_conv = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)  # 크기 조정
    img_conv = cv2.flip(img_conv, 0)
    img_conv = cv2.flip(img_conv, 1)

    #2. 색공간 필터링 - hls, hsv
    #색공간 변경 HLS로 white 검출, HSV로 노란색 차선 검출
    hls = cv2.cvtColor(img_conv, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img_conv, cv2.COLOR_BGR2HSV)

    if mode == 0 or mode == 1 or mode == 2 : 
        if mode == 0  or mode == 2 : 
            # 흰색 차선 (HLS) - 높은 L = 흰색
            lower_white = np.array(lower_white)
            upper_white = np.array(upper_white)
            mask_white = cv2.inRange(hls, lower_white, upper_white)

        if mode == 0 or mode == 1 : 
            # 노란색 차선 (HSV) - 10~40의 H 노란색
            lower_yellow = np.array(lower_yellow)
            upper_yellow = np.array(upper_yellow)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        if mode == 0 : img_conv = cv2.bitwise_or(mask_white,mask_yellow)
        elif mode == 1 : img_conv = mask_yellow
        elif mode == 2 : img_conv = mask_white

    else :
        return img_conv

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


def find_base(img, basefindgrid_N = 20, find_base_width_idx = 4, show = False):

    H,W = img.shape

    # 파라미터 가드
    if basefindgrid_N < 2 : 
        basefindgrid_N = 2
    if find_base_width_idx < 1 : 
        find_base_width_idx = 1
    
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


    left_max_idx = np.argmax(findbase_score[:left_th_idx]) # 0 ~ basefindgrid_midpoint-1 까지 검사
    left_max_idx = int(left_max_idx)

    right_max_idx = np.argmax(findbase_score[right_th_idx:]) # basefindgrid_midpoint ~ 끝까지 검사
    right_max_idx = right_th_idx + right_max_idx
    right_max_idx = int(right_max_idx)


    x1,x2 = findbase_edge[left_max_idx]
    leftx_base = (x1+x2)//2

    x1,x2 = findbase_edge[right_max_idx]
    rightx_base = (x1+x2)//2

    return leftx_base, rightx_base


def find_line_point(img,leftx_base,rightx_base, nwindows = 20, h_margin = 60, l_margin = 80, minpix = 20):
    """
    nwindows
    h_margin
    l_margin
    minpix

    """
    # 슬라이딩 윈도우 설정

    window_height = img.shape[0] // nwindows

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
        y_current = (win_y_low + win_y_high)/2

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
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            left_box_centor.append((leftx_current,y_current) )
        else :
            left_box_centor.append(None)

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            right_box_centor.append((rightx_current,y_current)) 
        else :
            right_box_centor.append(None)


    left_box_clean = [p for p in left_box_centor if p is not None]
    right_box_clean = [p for p in right_box_centor if p is not None]

    left_mean_x  = np.mean([x for (x, _) in left_box_clean])  if len(left_box_clean)  > 0 else None
    right_mean_x = np.mean([x for (x, _) in right_box_clean]) if len(right_box_clean) > 0 else None


    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([])
    left_pixel_count = len(left_lane_inds)
    
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([])
    right_pixel_count = len(right_lane_inds)


    return left_mean_x, right_mean_x, left_pixel_count, right_pixel_count , left_box_clean , right_box_clean


def debug_show_line_point(
    img,
    left_box_clean,   # [(x,y), (x,y), ...]
    right_box_clean,  # [(x,y), (x,y), ...]
    left_color=(0, 0, 255),   # BGR: 빨강 (왼쪽 라인)
    right_color=(255, 0, 0),  # BGR: 파랑 (오른쪽 라인)
    radius=4,
    thickness=-1,
    copy=True
):
    """
    left_box_clean / right_box_clean:
        find_line_point()에서 None 제거된 (x,y) 좌표 리스트
        y는 슬라이딩 윈도우의 세로중심, x는 해당 윈도우에서 찾은 차선 평균 x

    return:
        vis (BGR 디버그용 이미지)
    """

    # img 복사 & BGR 변환
    vis = img.copy() if copy else img
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 왼쪽 라인 점들 찍기
    for (cx, cy) in left_box_clean:
        cv2.circle(
            vis,
            (int(cx), int(cy)),
            radius,
            left_color,
            thickness,
            lineType=cv2.LINE_AA
        )

    # 오른쪽 라인 점들 찍기
    for (cx, cy) in right_box_clean:
        cv2.circle(
            vis,
            (int(cx), int(cy)),
            radius,
            right_color,
            thickness,
            lineType=cv2.LINE_AA
        )

    # 디버그 텍스트
    text = f"L:{len(left_box_clean)}  R:{len(right_box_clean)}"
    cv2.putText(
        vis,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 220, 50),
        2,
        cv2.LINE_AA
    )

    return vis


def find_line_fit(L_current, R_current, L_score , R_score, line_width = 200, flag = 0, img_mid = 250, bias = 50):
    x_location = None
    current_line = "MID"


    if (flag != 0 and flag != 1 and flag != 2):
        flag = 0

    if (flag == 0): # 일반 주행 모드
        if (R_score >= L_score and (R_current is not None)):
            current_line = "RIGHT"
            x_location = R_current - line_width

        elif (R_score <= L_score and (L_current is not None)) :
            current_line = "LEFT"
            x_location = L_current + line_width

        else :
            current_line = "NO LANE"
            x_location = None
    
    elif (flag == 1) : # 왼쪽 차선만 따라가는 모드
        if (L_current is not None):
            current_line = "LEFT"
            x_location = L_current + line_width

        else : 
            x_location = float(img_mid) - float(bias) 
            current_line = "NO LANE / / SEARCH LEFT"
    
    elif (flag == 2) : # 우측 차선만 따라가는 모드
        if (R_current is not None):
            current_line = "RIGHT"
            x_location = R_current - line_width

        else : 
            x_location = float(img_mid) + float(bias) 
            current_line = "NO LANE / / SEARCH RIGHT"
    
    return x_location , current_line
       

def debug_show_line_fit_points(img, x_location, l_current, r_current, current_line, y = 340):

    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    H, W = vis.shape[:2]

    # 화면 정중앙(카메라 기준)
    center_x = W // 2
    cv2.line(vis, (center_x, 0), (center_x, H), (255, 255, 0), 1)  # cyan center ref

    # 목표 주행 위치 (초록)
    if x_location is not None:
        cv2.circle(vis, (int(x_location), y), 5, (0, 255, 0), -1)
        cv2.putText(vis, f"target:{int(x_location)}", (int(x_location) - 40, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 왼쪽 차선 추정점 (빨강)
    if l_current is not None:
        cv2.circle(vis, (int(l_current), y), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"L:{int(l_current)}", (int(l_current) - 40, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 오른쪽 차선 추정점 (파랑)
    if r_current is not None:
        cv2.circle(vis, (int(r_current), y), 5, (255, 0, 0), -1)
        cv2.putText(vis, f"R:{int(r_current)}", (int(r_current) - 40, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 현재 어떤 라인 따라가는지
    cv2.putText(vis, f"LINE: {current_line}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def find_angle(img, x_location , last_angle = 0.0 , Kp = 0.4, max_deg = 36.0):
    H, W = img.shape[:2]      # ← 이렇게 바꾸면 흑백/컬러 둘 다 안전

    img_center_x = W / 2.0

    if x_location is None:
        # 차선 못 본 프레임이면 이전 각 유지
        return last_angle

    e = float(x_location) - float(img_center_x)

    angle_deg = Kp * e
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


class Driver:
    def __init__(self):
        ######################################전역변수

        self.past_L_base_idx = None
        self.past_R_base_idx = None
        self.prev_time = None
        self.last_angle = 0.0

    def pre_process(self,
                    img, 
                    mode = 0, 
                    resize = (640, 480), 
                    roi = [[100, 260], [540, 260], [640, 420], [0, 420]], 
                    bev_size = (500, 420), 
                    lower_white = [0, 230, 0], upper_white = [180, 255, 255],
                    lower_yellow= [0,0,200], upper_yellow= [179,255,255],
                    show = False
                    ) : 
        """
        mode 0 흰색 노란색
        mode 1 노란색
        mode 2 흰색
        """

        debug_show = [] 

        if (show == True ) :
            img_show = img.copy()
            img_show = cv2.flip(img_show, 0)
            img_show = cv2.flip(img_show, 1)
            debug_show.append(img_show) #show point1 : ori img

        mask = pre_process_1(img,
                            lower_white = lower_white,
                            upper_white = upper_white,
                            lower_yellow = lower_yellow,
                            upper_yellow = upper_yellow,
                            resize = resize,
                            mode = mode
                            )
        
        p1 = mask
                            
        if (show == True ) : debug_show.append(debug_draw_roi(mask,roi_list = roi)) #show point2 : coler filtering and roi 

        p2  = pre_process_2(mask,
                            roi = roi,
                            bev_size = bev_size
                            )
        
        if (show == True ) : debug_show.append(p2)

        return p2, p1, debug_show


    def drive(self, img, flag = 0 , show = False , line_width = 200 , bias = 50 , max_deg = 36 , find_base_width_idx = 4) :
        debug_show = []

        # 1) 베이스 x 찾기
        left_base, right_base = find_base(
            img,
            basefindgrid_N = 20,
            find_base_width_idx = find_base_width_idx,
            show = False
        )

        # 2) 슬라이딩 윈도우로 왼/오 차선 평균 x랑 픽셀 스코어 얻기
        l_current, r_current, l_score , r_score , l_box , r_box = find_line_point(
            img,
            left_base,
            right_base
        )

        if show:
            debug_show.append(
                debug_show_line_point(
                    img,
                    l_box,
                    r_box
                )
            )

        # 3) 실제 주행 타겟 x_location 계산
        x_location, current_line = find_line_fit(
            l_current,
            r_current,
            l_score,
            r_score,
            line_width = line_width,
            flag = flag,
            bias = bias
        )

        if show:
            debug_show.append(
                debug_show_line_fit_points(
                    img,
                    x_location,
                    l_current,
                    r_current,
                    current_line
                )
            )

        # 4) 조향각 계산 (라인 없으면 self.last_angle 그대로 유지)
        angle = find_angle(
            img,
            x_location,
            last_angle = self.last_angle,  # ← 여기 중요
            max_deg = max_deg
        )

        # 5) 다음 프레임을 위한 상태 업데이트
        self.last_angle = angle

        # 6) 디버그용 조향 화살표
        if show:
            debug_show.append(
                draw_steering_angle(
                    img,
                    angle,
                    length=230,
                    color=(0, 255, 255)
                )
            )

        return angle, current_line ,debug_show

    def is_stop_line(self, mask_white, roi_yp = 70, roi_h_p = 30, roi_w_p = 100, show=False):

        # [STEP 1] 기본 정보
        H, W = mask_white.shape[:2]

        # [STEP 2] ROI 자동 계산 (중앙 대칭)
        roi_h = int(H * roi_h_p / 100.0)
        roi_y = int(H * roi_yp / 100.0)
        roi_w = int(W * roi_w_p / 100.0)
        roi_x = (W - roi_w) // 2

        # 안전 클리핑 (혹시라도 비율 계산 오류 방지)
        roi_y = np.clip(roi_y, 0, H - 1)
        roi_h = np.clip(roi_h, 1, H - roi_y)
        roi_x = np.clip(roi_x, 0, W - 1)
        roi_w = np.clip(roi_w, 1, W - roi_x)

        # [STEP 3] ROI 영역만 추출
        roi_mask = mask_white[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # [STEP 4] 컨투어 검출 (뭉쳐 있는 흰색 영역)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 시각화용 (디버그 이미지)
        if (show == True):
            debugimg = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
        else:
            debugimg = None

        found = False

        # [STEP 5] 컨투어 기반 정지선 판별
        for cnt in contours:

            if cv2.contourArea(cnt) < 50:  # 너무 작은 노이즈 제거
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            # 너무 작은 사각형은 패스(노이즈)
            if cw < 50 and ch < 50:
                continue
            is_wide = cw >= int(roi_w * 0.5)
            is_thin = ch >= int(roi_h * 0.1)
            is_horizontal = (cw > ch)  # 가로가 세로보다 긴지
            if is_wide and is_thin and is_horizontal :
                found = True
                if (show == True):
                    cv2.rectangle(debugimg, (roi_x + cx, roi_y + cy),
                                  (roi_x + cx + cw, roi_y + cy + ch),
                                  (0, 0, 255), 2)
                    cv2.putText(debugimg, "STOP LINE!!",
                                (roi_x + cx, max(0, roi_y + cy - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else :
                if (show == True):
                    cv2.rectangle(debugimg, (roi_x + cx, roi_y + cy),
                                  (roi_x + cx + cw, roi_y + cy + ch),
                                  (0, 255, 255), 2)
                    cv2.putText(debugimg, "STOP LINE??",
                                (roi_x + cx, max(0, roi_y + cy - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # [STEP 6] ROI 표시 및 결과 상태 텍스트
        roi_color = (255, 255, 255)

        if (show == True):
            cv2.rectangle(debugimg, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), roi_color, 2)
            cv2.putText(debugimg, f"STOP LINE: {'FOUND' if found else 'SEARCHING'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if found else (0, 255, 0), 2)

        return found, debugimg


    def is_stop_line2(self, mask_white, roi_y_p = 70, roi_h_p = 30, roi_w_p = 100, max_tilt = 45 , show = False):

        # [STEP 1] mask_white 크기 가져오기
        H, W = mask_white.shape[:2]

        # [STEP 2] ROI 자동 계산 (중앙 대칭)
        roi_h = int(H * roi_h_p / 100.0)
        roi_y = int(H * roi_y_p / 100.0)
        roi_w = int(W * roi_w_p / 100.0)
        roi_x = (W - roi_w) // 2

        # 안전 클리핑 (혹시라도 비율 계산 오류 방지)
        roi_y = np.clip(roi_y, 0, H-1)
        roi_h = np.clip(roi_h, 1, H - roi_y)
        roi_x = np.clip(roi_x, 0, W-1)
        roi_w = np.clip(roi_w, 1, W - roi_x)

        # [STEP 3] ROI 영역만 추출
        roi_mask = mask_white[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # [STEP 4] 컨투어 검출 (뭉쳐 있는 흰색 영역)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 시각화용 (디버그 이미지)
        if show == True : debugimg = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
        else : debugimg = mask_white
        found = False

        # --- ROI 기준 참조값 ---
        roi_diag    = math.hypot(roi_w, roi_h)                # ROI의 대각선 길이
        len_thr_px  = roi_diag * 0.2                   ### 인식할 너비 기준 (대각선의 80%이상) ###
        thick_thr_px= roi_h * 0.1                     ### 인식할 높이 기준 (세로의 30%이상) ###

        # [STEP 5] 컨투어 기반 정지선 판별
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:   # 너무 작은 노이즈 제거
                continue

            # 객체 윤곽선을 외접하는 가장 작은 직사각형 반환
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt) # ((cx,cy),(w_rot,h_rot), angle)로 반환
                                                            # (직사각형 중심, 폭, 높이, 수평선으로부터 기울어진 각도)

            # 너무 작은 사각형은 패스(노이즈)
            if rw < 1 or rh < 1:
                continue

            # 각도 -90~0인 것을 0~90으로 정규화
            tilt = abs(angle) if rw >= rh else abs(angle + 90)  # 0~90도 범위로 정규화

            line_len    = max(rw, rh)               # 장축 = 라인의 길이
            line_thick  = min(rw, rh)               # 단축 = 라인의 두께
            is_wide = line_len >= len_thr_px        # 기준 너비보다 객체의 너비가 넓은지
            is_thick = line_thick >= thick_thr_px   # 기준 두께보다 두께의 너비가 두꺼운지

            # --- 정지선 판단 조건 (수평 정지선 외에 기울여져서 인식하는 경우를 방어)
            # 1) 길이가 충분히 길고(ROI 대각선 기준), 2) 두께도 충분, 3) 수평 정지선 기준 60도 틀어진것까지만 인정
            if is_wide and is_thick and (tilt <= max_tilt):
                # 시각화 (전체 좌표로 변환)
                rect = cv2.minAreaRect(cnt)                  # cnt: 단일 컨투어 점들
                box  = cv2.boxPoints(rect).astype(int)       # 4점 좌표 (4,2)
                box[:,0] += roi_x; box[:,1] += roi_y
                cv2.drawContours(debugimg, [box], 0, (0, 255, 255), 2)
                cv2.putText(debugimg, "STOP LINE", (roi_x + int(cx), roi_y + max(0, int(cy) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                found = True
                #break  # 하나만 찾으면 충분하면 break

        # [STEP 6] ROI 표시 및 결과 상태 텍스트
        roi_color = (0, 255, 0) if found else (255, 255, 255)
        cv2.rectangle(debugimg, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), roi_color, 2)
        cv2.putText(debugimg, f"STOP LINE: {'FOUND' if found else 'SEARCHING'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2)

        return found, debugimg


if __name__ == "__main__" : 
    pass