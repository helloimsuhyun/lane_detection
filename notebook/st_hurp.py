import cv2
import numpy as np
import os
import math
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

def pre_process_2(img, roi = [[70, 200],[570, 200],[640, 480],[0, 480]], bev_size = (500,420)):
    """
    bird eye view 변환
    """
    # bird's eye view를 적용할 roi를 넘파이 배열로 지정
    src = np.array(roi, dtype=np.float32) #좌상단 우상단 우하단 좌하단
    width, height = bev_size[0], bev_size[1]
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


def find_Right_line_1(img, past_xR, step_w ,search_upper,search_lower, debug = False):
    H, W = img.shape

    if past_xR == None : 
        past_xR = W // 2
    
    i = 1

    while(1):
        if (past_xR + i* step_w > W) : break

        search_roi = (past_xR - step_w ,search_upper, past_xR + i * step_w ,search_lower)
        avg_angle , past_point, find_roi, lines = find_Right_line_2(img,search_roi, W//2, debug = True)
        if avg_angle is None or past_point is None : 
            i += 1
            continue
        else :
            return avg_angle, past_point, find_roi, lines # 직선 검출 > 탈출
        
    return None, None, None, None


def find_Right_line_2(img,search_roi, img_centor = 320, debug = False):
    """
    y1 x1 좌상단
    y2 x2 우하단
    """
    x1,y1,x2,y2 = search_roi
    line_img = img[y1:y2, x1:x2]

    # 1. 허프 파라미터 설정
    rho = 1                # 거리 해상도 (픽셀)
    theta = np.pi / 180    # 각도 해상도 (1도)
    threshold = 30         # 누적 투표 임계값
    minLineLength = 10    # 최소 선분 길이
    maxLineGap = 200        # 선분 연결 허용 간격

    # 2. 허프 변환 실행
    lines = cv2.HoughLinesP(
        line_img,
        rho,
        theta,
        threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )

    #3. 라인 탐색 
    if lines is not None:
        lines = lines.reshape(-1,4) # n행 4열, x1,y1 x2 y2
        dx = lines[:,2] - lines[:,0] # x2 - x1
        dy = lines[:,3] - lines[:,1] # y2 - y1
        angles = 90 - np.degrees(np.arctan2(dy, dx))
        avg_angle = np.mean(angles)

        
        y_value = np.concatenate((lines[:,3],lines[:,1]))
        x_value = np.concatenate((lines[:,2],lines[:,0]))
        
        
        past_y = y1 + y_value[np.argmin(y_value)]
        past_x = x1 + x_value[np.argmin(x_value)]
        
        roi_list=[[x1, y1], [x2, y1], [x2,y2], [x1, y2]]
        lines = lines

        return avg_angle, (past_x, past_y), roi_list, lines # 검출직선의 평균각, 검출된 직선 중 제일 좌상단 점, 검출된 roi list, 검출된 선분 전부 반환

    else : 
        return None, None, None, None


def debug_find_line(img, roi_list , past_point, past_deg, lines):
    h,w = img.shape
    out = img.copy()

    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if (roi_list != None and past_point != None and past_deg != None) : 
        #find roi 그리기
        pts = np.array(roi_list, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        local_y_base = roi_list[0][1]
        local_x_base = roi_list[0][0]

        y_value = np.concatenate((lines[:,3],lines[:,1]))
        x_value = np.concatenate((lines[:,2],lines[:,0]))

        y_max = np.max(y_value)

        # ② y가 최대인 점들의 인덱스 찾기
        idx_y_max = np.where(y_value == y_max)[0]

        # ③ 그 중 x가 최소인 점 선택
        best_idx = idx_y_max[np.argmin(x_value[idx_y_max])]

        # ④ 글로벌 좌표 계산
        x_2 = local_x_base + x_value[best_idx]
        y_2 = local_y_base + y_value[best_idx]
        
        past_x, past_y = past_point

        cv2.line(out, (int(past_x), int(past_y)), (int(x_2), int(y_2)), (255, 0, 0), 2)
        """
        for x1, y1, x2, y2 in lines: #검출된 모든 선분 그리기
            cv2.line(out, (int(x1+local_x_base), int(y1+local_y_base)), (int(x2+local_x_base), int(y2+local_y_base)), (0, 255, 0), 2)
        """

        #find 직선의 좌상단 점 찍기
        past_x, past_y = past_point
        cv2.circle(out, (int(past_x), int(past_y)), 6, (0, 0, 255), -1)

    return out


def find_base(img, roi, basefindgrid_N = 20):

    """
    1. img : 받아올 원본 이미지
    2. roi array : 슬라이딩 윈도우를 진행할 roi [[x1,y1], [x2,y1] , [x2,y2], [x1,y2]] , 좌상단 우상단 우하단 좌하단
    3. basefindgrid_N : basefind를 할 gird 갯수

    return 슬라이딩 윈도우 베이스
    """
    xm = roi[0][0]
    ym = roi[0][1]
    xM = roi[2][0]
    yM = roi[2][1]

    line_img = img[ym:yM, xm:xM]
    H,W = line_img.shape
    
    basefindgrid_margin = W // basefindgrid_N

    findbase_score = []
    findbase_edge = []

    for i in range(basefindgrid_N) :
        x1 = i * basefindgrid_margin
        x2 = W if i == basefindgrid_N -1 else (i+1)*basefindgrid_margin # 마지막 grid일 경우 대비 보정항
        
        temp = np.sum(line_img[:, x1:x2])
        findbase_score.append(temp)
        findbase_edge.append((x1,x2))

    findbase_score = np.array(findbase_score)

    max_idx = np.argmax(findbase_score[:])
    max_idx = int(max_idx)

    x1,x2 = findbase_edge[max_idx]
    base_local = (x1+x2)//2
    base_global = xm + base_local

    return base_global, base_local

def find_line_point(img, base_global, roi ,nwindows = 20, minpix = 10):
    """
    nwindows
    h_margin
    l_margin
    minpix
    """

    # 슬라이딩 윈도우 설정
    xm = roi[0][0]
    ym = roi[0][1]
    xM = roi[2][0]
    yM = roi[2][1]

    l_margin = (abs(xm - xM))//2
    h_margin = (abs(xm - xM))//3

    img = img[ym:yM, xm:xM]
    base_local = base_global - xm

    window_height = img.shape[0] // nwindows
    box_current = base_local

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    line_box_centor = []

    for window in range(nwindows):
        
        if (window <= 2 and window >= 0): 
            margin = l_margin
        else : 
            margin = h_margin
            
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height

        win_xleft_low = box_current - margin
        win_xleft_high = box_current + margin

        # 윈도우 내 픽셀 인덱스 추출
        lane_pix = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]


        # 윈도우의 무게중심 위치를 centor 변수에 저장, 리스트에 넣음

        if len(lane_pix) > minpix :
            x_centor = int(np.mean(nonzerox[lane_pix]))
            box_current = x_centor
        else :
            x_centor = None
            
        y_centor = (win_y_low+win_y_high)//2

        if x_centor != None : 
            global_x_centor = x_centor + xm
            global_y_centor = y_centor + ym
        else : 
            global_x_centor = None
            global_y_centor = y_centor + ym 


        
        line_box_centor.append((global_x_centor,global_y_centor))

    line_box_clean = [p for p in line_box_centor if p[0] is not None]
 
    return line_box_clean

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
























past_xR = None

if __name__ == "__main__" : 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir,'data/line_avi_dataset/data/videos/train','02.avi')

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 끝까지 읽으면 종료
        mask = pre_process_1(frame)  # 전처리1 색상필터링/가우시안블러
        roi = [[70, 350], [570, 350], [640, 480], [0, 480]]
        bev  = pre_process_2(mask, roi, bev_size = (400,300))     # BEV + Canny
        show1 = debug_draw_roi(frame,roi)

        
        bev_h , bev_w = bev.shape
        avg_angle, past_point, find_roi, lines = find_Right_line_1(img = bev, past_xR = past_xR, step_w = 10 , search_upper = 0 ,search_lower = bev_h, debug = True)

        show2 = debug_find_line(bev,find_roi,past_point,avg_angle,lines)
        if past_point == None :
            past_xR = None
        else : 
            past_xR = past_point[0]
            if (past_xR <= bev_w //2) : 
                past_xR = None
        
        if find_roi != None : 
            right_base, _ = find_base(bev,find_roi)
            right_line_point = find_line_point(bev, right_base,find_roi)
            show3 = debug_show_line_point(bev, right_line_point,right_line_point )
            cv2.imshow("find line point", show3)    

        cv2.imshow("debug_bev_roi", show1)
        cv2.imshow("debug_find_line", show2)    


        # ESC나 q 누르면 종료
        if cv2.waitKey(50) & 0xFF in [27, ord('q')]:
            break
    
    print("end")
    cap.release()
    cv2.destroyAllWindows()
    

