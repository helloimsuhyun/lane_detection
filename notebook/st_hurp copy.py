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
        avg_line , past_point, find_roi, lines = find_Right_line_2(img,search_roi, W//2, debug = True)
        if avg_line is None or past_point is None or find_roi is None or lines is None : 
            i += 1
            continue
        else :
            return avg_line, past_point, find_roi, lines # 직선 검출 > 탈출
        
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

    x_sum = 0
    y_sum = 0
    m_sum = 0
    cnt = 0

    #3. 라인 탐색 
    if lines is not None:
        lines = lines.reshape(-1,4) # n행 4열, x1,y1 x2 y2
        num = len(lines)

        X1 = lines[:,0] + x1 
        X2 = lines[:,2] + x1 
        Y1 = lines[:,1] + y1 
        Y2 = lines[:,3] + y1 

        dx = lines[:,2] - lines[:,0] # x2 - x1
        dy = lines[:,3] - lines[:,1] # y2 - y1

        for i in range(num) :
            if (X2[i] - X1[i] != 0) :
                x_sum += X1[i] + X2[i]
                y_sum += Y1[i] + Y2[i]
                m_sum += float(Y2[i] - Y1[i]) / float (X2[i] - X1[i])
                cnt += 1

        x_avg = x_sum / (cnt*2)
        y_avg = y_sum / (cnt*2)
        m_avg = m_sum / cnt
        b_avg = y_avg - m_avg * x_avg

        y_value = np.concatenate((lines[:,3],lines[:,1]))
        x_value = np.concatenate((lines[:,2],lines[:,0]))
        
        past_y = y1 + y_value[np.argmin(y_value)]
        past_x = x1 + x_value[np.argmin(x_value)]
        
        roi_list=[[x1, y1], [x2, y1], [x2,y2], [x1, y2]]
        lines = lines

    

        return (m_avg,b_avg), (past_x, past_y), roi_list, lines # 검출직선의 평균각, 검출된 직선 중 제일 좌상단 점, 검출된 roi list, 검출된 선분 전부 반환

    else : 
        return None, None, None, None


def debug_find_line(img, roi_list , past_point, avg_line, lines): 

    h,w = img.shape
    out = img.copy()

    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if (roi_list != None and past_point != None and avg_line != None) : 
        m,b = avg_line

        #find roi 그리기
        pts = np.array(roi_list, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        find_x_base = roi_list[0][0]
        find_y_base = roi_list[0][1]

        find_x_bottom = roi_list[2][0]
        find_y_bottom = roi_list[2][1]

        if np.isfinite(m) and np.isfinite(b):

            avg_y1 = m * find_x_base + b 
            avg_y2 = m * find_x_bottom + b

            if (np.isfinite(avg_y1) and np.isfinite(avg_y2)) : 
                x1 = find_x_base
                y1 = avg_y1
                x2 = find_x_bottom
                y2 = avg_y2

            else :
                y1 = find_y_base
                y2 = find_y_bottom
                x1 = (find_x_base + find_x_bottom) / 2
                x2 = (find_x_base + find_x_bottom) / 2

        elif np.isinf(m) and np.isfinite(b):
        # 수직선 x=b
            y1 = find_y_base
            y2 = find_y_bottom
            x1 = (find_x_base + find_x_bottom) / 2
            x2 = (find_x_base + find_x_bottom) / 2

        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        #for x1, y1, x2, y2 in lines: #검출된 모든 선분 그리기
        #    cv2.line(out, (int(x1+find_x_base), int(y1+find_y_base)), (int(x2+find_x_base), int(y2+find_y_base)), (0, 255, 0), 2)
        

        #find 직선의 좌상단 점 찍기
        past_x, past_y = past_point
        cv2.circle(out, (int(past_x), int(past_y)), 6, (0, 0, 255), -1)

    return out




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
        avg_line, past_point, find_roi, lines = find_Right_line_1(img = bev, past_xR = past_xR, step_w = 40 , search_upper = 0 ,search_lower = bev_h, debug = True)

        show2 = debug_find_line(bev,find_roi,past_point,avg_line,lines)
        if past_point == None :
            past_xR = None
        else : 
            past_xR = past_point[0]
            if (past_xR <= bev_w //2) : 
                past_xR = None
        
        cv2.imshow("debug_bev_roi", show1)
        cv2.imshow("debug_find_line", show2)    


        # ESC나 q 누르면 종료
        if cv2.waitKey(50) & 0xFF in [27, ord('q')]:
            break
    
    print("end")
    cap.release()
    cv2.destroyAllWindows()
    

