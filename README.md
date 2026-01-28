# Vision-based Lane Detection & Steering

본 프로젝트는 **카메라 영상 기반 차선 인식 및 조향 제어(line following)**를 위한  
슬라이딩 윈도우 기반 비전 모듈이다.

OpenCV 기반 전처리와 sliding window 방식으로 좌·우 차선을 추적하고,  
차선 중심 오차를 이용해 **steering angle**을 계산한다.

---

## 🎥 Demo Video

[![Lane Following Demo](https://img.youtube.com/vi/u-K3Kae270E/0.jpg)](https://youtu.be/u-K3Kae270E)

---

## 주요 기능

- Sliding window 기반 lane detection
- 좌·우 차선 인식 및 중심선 추정
- 차선 오차 기반 steering angle 계산
- Kalman filter 기반 steering smoothing
- 디버그용 시각화 (lane point, target x, steering angle)

---

## 기본 사용 예시

카메라 또는 영상 프레임(`frame`, BGR 이미지)을 입력으로 사용한다.

```python
driver = Driver()

# 전처리 (BEV edge 이미지 생성)
bev_edge, mask, _ = driver.pre_process(frame)

# 차선 추종 및 조향각 계산
angle, line_state, _ = driver.drive(bev_edge)
```

angle : 계산된 조향각 (deg)

line_state : 현재 추종 중인 차선 상태 (LEFT, RIGHT, NO LANE)
