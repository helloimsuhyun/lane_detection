
from drive2 import Driver
import os
import cv2

drv = Driver()
debug = True

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir,'IRO_example4.avi')
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 끝까지 읽으면 종료
        
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        #stop_line 디버깅
        _ , stop_pre_img , show = drv.pre_process(img = frame , mode = 2, lower_white = [0, 230, 0], upper_white = [180, 255, 255] , show = debug)
        stop_line, show_  = drv.is_stop_line(stop_pre_img,
                                            show = debug)
        cv2.imshow("original image", show[0])
        cv2.imshow("white filter img", stop_pre_img)
        cv2.imshow("stop_line_detect", show_)

        #차선인식 디버깅
        bev , _ , show = drv.pre_process(img = frame , mode = 1,
                                         roi = [[100, 260], [540, 260], [640, 420], [0, 420]],
                                         lower_yellow= [0,0,200],
                                         upper_yellow= [179,255,255],
                                         bev_size = (500, 420),
                                         show = debug)
        cv2.imshow("yellow_fliter_and_roi", show[1])
        cv2.imshow("bev_img", show[2])


        angle, current_line , show = drv.drive(img = bev, flag = 0 ,line_width = 200 , max_deg = 36,  show = debug) # 양쪽차선
        #angle, current_line, show = drv.drive(img = bev, flag = 1 ,line_width = 195 , bias = 50 , max_deg = 36,, show = debug) #좌차선
        #angle, current_line, show = drv.drive(img = bev, flag = 2 ,line_width = 195 , bias = 50, max_deg = 36, show = debug) #우차선
        cv2.imshow("currddddddddnet_x", show[0])
        cv2.imshow("currnet_x", show[1])
        cv2.imshow("drive_angle", show[2])
        
        # ESC나 q 누르면 종료
        if cv2.waitKey(50) & 0xFF in [27, ord('q')]:
            break
    
    print("end")
    cap.release()
    cv2.destroyAllWindows()

