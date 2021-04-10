# MediaPipe Holistic (Image)

2021.04.11

[MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)

### 01. Whole Code

```python
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# For static images:
path = "./images/"
file_list = os.listdir(path)

with mp_holistic.Holistic(static_image_mode=True) as holistic:
  for idx, file in enumerate(file_list):
    image = cv2.imread(path+file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw pose, left and right hands, and face landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Use mp_holistic.UPPER_BODY_POSE_CONNECTIONS for drawing below when
    # upper_body_only is set to True.
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    cv2.imwrite('./results/result' + str(idx) + '.jpg', annotated_image)
cv2.imshow("MediaPipe Holistic", annotated_image)
```

<br>

###  02. MediaPipe Holistic Image Result

1. image0.jpg & result0.jpg

   ![result0.jpg](https://github.com/hyunmin0317/OpenCV_Study/blob/master/MediaPipePose/MediaPipeHolistic(Image)/Github/result0.jpg?raw=true)

2. image1.jpg & result1.jpg

   ![result1.jpg](https://github.com/hyunmin0317/OpenCV_Study/blob/master/MediaPipePose/MediaPipeHolistic(Image)/Github/result1.jpg?raw=true)

3. image2.jpg & result2.jpg

   ![result2.jpg](https://github.com/hyunmin0317/OpenCV_Study/blob/master/MediaPipePose/MediaPipeHolistic(Image)/Github/result2.jpg?raw=true)



