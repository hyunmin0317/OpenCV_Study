import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For static images:
path = "./images/"
file_list = os.listdir(path)

with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(file_list):
    image = cv2.imread(path+file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue

    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
    # upper_body_only is set to True.
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite('./results/result' + str(idx) + '.jpg', annotated_image)
cv2.imshow("MediaPipe Pose", annotated_image)
