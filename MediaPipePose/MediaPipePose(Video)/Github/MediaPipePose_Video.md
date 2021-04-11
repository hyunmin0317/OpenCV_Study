# MediaPipe Pose (Video)

2021.04.09

[MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)

### 01. Pose Landmark Model (BlazePose GHUM 3D)

![image01.PNG](https://github.com/hyunmin0317/OpenCV_Study/blob/master/MediaPipePose/MediaPipePose(Video)/Github/image01.PNG?raw=true)

<br>

### 02. Solution APIs

* **Cross-platform Configuration Options**
  * MIN_DETECTION_CONFIDENCE
    * 탐지가 성공한 것으로 간주되는 사람 탐지 모델의 최소 신뢰 값 ([0.0, 1.0])으로 기본값은 0.5
  * MIN_TRACKING_CONFIDENCE
    * pose landmarks 를 성공적으로 추적하는 것으로 간주되는 랜드마크 추적 모델의 최소 신뢰 값 ([0.0, 1.0])으로 기본값은 0.5
    * 반면에 추적에 실패하면 사람 탐지 모델은 다음 입력 이미지에서 자동으로 호출하고 모든 이미지에서 실행됨

* **POSE_LANDMARKS Attribute**
  * `x` and `y`: 이미지 너비와 높이를 통해 `[0.0, 1.0]` 로  정규화 된 landmark 좌표
  * `z`: 골반의 중간 점 깊이를 원점으로하는 landmark 깊이
    * `z` 값이 작을수록 랜드 마크가 카메라에 가까워짐
    * `z` 의 크기는 `x` 와 거의 같은 scale을 사용함
    * 참고: `z`는 전신 모드에서만 예측되며 `upper_body_only` 가 `true` 인 경우 삭제해야함
  * `visibility`: 이미지에서 랜드마크가 영상에 표시될 가능성을 나타내는 값으로 `[0.0, 1.0]` 의 값을 갖음

<br>

### 03. Python Solution API

```python
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

<br>

###  04. MediaPipe Pose Video Result

![result.PNG](https://github.com/hyunmin0317/OpenCV_Study/blob/master/MediaPipePose/MediaPipePose(Video)/Github/result.PNG?raw=true)

