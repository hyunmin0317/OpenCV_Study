# [OpenCV Age Detection with Deep Learning]((https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/))

2021.03.22

<br>

### 01. What is age detection?

* Age detection: 얼굴 사진 만으로 사람의 나이를 자동으로 알아보는 과정

* Age detection 2-stage process

  1. input image/video stream 에서 얼굴 감지

     * 이미지 중 얼굴에 대한 bounding boxes를 생성하기 위해 face detector 사용

     * face detector 종류 (프로젝트 요구사항에 맞춰 알맞은 detector를 선택해야함)

       * Haar cascades: 매우 빠르고 임베디드 기기에서 실시간으로 실행될 수 있지만 정확성이 떨어지고 false-positive detections이 발생하기 쉬운 문제가 있음

       * HOG + Linear SVM: Haar cascades 보다 정확하지만 느리고 occlusion과 viewpoint changes에 유연하지 않음

       * ##### Deep learning-based face detectors: 가장 정확하지만 훨씬 많은 계산 resources를 필요로 함

  2. 얼굴 Region of Interest (ROI)를 추출하고 age detector algorithm을 통해 사람의 나이 예측

     * 이미지의 noise를 제외하고 사람의 얼굴에만 초점을 맞추기 위해  bounding box (x, y)를 지정함
     * 그 다음 얼굴 ROI를 추출하고 얼굴 ROI를 model에 통과시켜 나이 예측 결과를 산출함  

<br>

### 02. Our age detector deep learning model

<br>

### 03. Why aren’t we treating age prediction as a regression problem?

<br>

### 04. Project structure

<br>

### 05. Implementing our OpenCV age detector for images

<br>

### 06. OpenCV age detection results

<br>

### 07. Implementing our OpenCV age detector for images

<br>

### 08. Implementing our OpenCV age detector for real-time video streams

<br>

### 09. Real-time age detection with OpenCV results

<br>

### 10. How can I improve age prediction results?

<br>

### 11. What about gender prediction?

<br>

### 12. Do you want to train your own deep learning models?

<br>

### 13. Summary