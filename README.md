<div align="center">

# Vision-based Lane Detection & Steering

**OpenCV · Sliding Window · Steering Control**

카메라 영상 기반으로 차선을 인식하고,  
차선 중심 오차를 이용해 **steering angle**을 계산하는 line following 모듈입니다.

<br/>

<img src="https://img.shields.io/badge/OpenCV-Vision-blue">
<img src="https://img.shields.io/badge/Lane%20Detection-Sliding%20Window-green">
<img src="https://img.shields.io/badge/Control-Steering%20Angle-orange">
<img src="https://img.shields.io/badge/Filter-Kalman%20Filter-purple">

</div>

---

## 🎥 Demo

<div align="center">

<a href="https://youtu.be/u-K3Kae270E">
  <img src="https://img.youtube.com/vi/u-K3Kae270E/0.jpg" width="70%">
</a>

</div>

---

## 주요 기능

- Sliding window 기반 lane detection
- 좌·우 차선 인식 및 중심선 추정
- 차선 오차 기반 steering angle 계산
- Kalman filter 기반 steering smoothing
- 디버그용 시각화

---

## 기본 사용 예시

```python
driver = Driver()

# 전처리
bev_edge, mask, _ = driver.pre_process(frame)

# 차선 추종 및 조향각 계산
angle, line_state, _ = driver.drive(bev_edge)
