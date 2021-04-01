# OpenCV Age Detection with Deep Learning

2021.03.22 

[OpenCV Age Detection with Deep Learning](https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)

<br>

### 01. What is age detection?

![image01]()

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

![image02]()

* The deep learning age detector model 
  * 2015년에 Levi와 Hassner가 구현한 model로 8가지 연령층을 학습하는 단순화된 AlexNet 구조를 사용
  * 8가지 연령층: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100 -> 비연속적
  * 모델을 교육하는데 사용되는 Adience dataset에서 연령 범위를 정의하기 위해 age brackets을 비연속적으로 설정

<br>

### 03. Why aren’t we treating age prediction as a regression problem?

* 나이 예측은 외모에만 근거하며 주관적이라 예측하기 어렵다는 문제점이 있음
* 그래서 나이 예측을 회귀 문제로 다루면 model이 나이를 단일값으로 정확하게 예측하기 어려움
* 하지만 이를 분류 문제로 간주하여 model에 buckets/age brackets를 정의하면 age detection model을 훈련하기 쉬우며 상당히 높은 정확도를 산출하므로 이를 사용해 프로젝트 진행

<br>

### 04. Project structure

* 프로젝트의 directory는 딥러닝 model인 age predictor와 face detector 그리고 이미지 파일들인 images로 구성됨
* 여기서 다룰 2개의 Python scripts
  * detect_age.py : Age prediction in single image
  * detect_age_video.py : Age prediction in video streams
* 각 scripts는 image/frame에서 얼굴을 감지한 후 OpenCV를 사용하여 연령 예측 수행

<br>

### 05. Implementing our OpenCV age detector for images

* static images에서 OpenCV를 사용한 연령 예측 구현 (detect_age.py)

  * import와 command line arguments

    ```python
    # import the necessary packages
    import numpy as np
    import argparse
    import cv2
    import os
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image")
    ap.add_argument("-f", "--face", required=True,
    	help="path to face detector model directory")
    ap.add_argument("-a", "--age", required=True,
    	help="path to age detector model directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
    	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    ```
    * 프로젝트 시작을 위해 NumPy와 OpenCV를 import 하고 모델 경로를 기입하기 위해 파이썬의 built-in module인 os를 import 하며 command line 명령을 위해 argparse를 import 함
    * command line arguments
      * --image : age detection을 위한 input image의 경로 제공
      * --face : 사전에 훈련된 face detection model directory 경로
      * --age : 사전에 훈련된 age detection model directory
      * --confidence : weak detections을 필터링하기 위한 최소 확률 임계 값

  <br>

  * age detection이 예측할 연령 bucket 목록을 정의

    ```python
    # define the list of age buckets our age detector will predict
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
    	"(38-43)", "(48-53)", "(60-100)"]
    ```

    <br>

  * 사전에 학습된 2가지 model을(age detection, face detection) 로드

    ```python
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
    	"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # load our serialized age detector model from disk
    print("[INFO] loading age detector model...")
    prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
    weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
    ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    ```

    * face detector은 image에서 얼굴을 찾고 age detector는 찾은 얼굴이 속한 연령 범위 결정

  <br>

  * 디스크에서 image를 로드하고 얼굴 ROI를 감지

    ```python
    # load the input image and construct an input blob for the image
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
    	(104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()
    ```

    * --image에 의해 로드되며 OpenCV의 blobFromImage에 의해 image에서 얼굴을 감지하고 결과를 detections의 리스트로 blob에 저장

  <br>

  * 얼굴 ROI detections를 반복

    ```python
    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]
        
    	# filter out weak detections by ensuring the confidence is
    	# greater than the minimum confidence
    	if confidence > args["confidence"]:
    		# compute the (x, y)-coordinates of the bounding box for the
    		# object
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    
            # extract the ROI of the face and then construct a blob from
    		# *only* the face ROI
    		face = image[startY:endY, startX:endX]
    		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
    			(78.4263377603, 87.7689143
    ```

    * detections을 반복하며 bucket에 대한 확률을 confidence에 저장하며 최소 기준을 충족하는 얼굴의 경우 ROI 좌표를 추출
    * 계속해서 ROI를 반복하며 얼굴만 포함된 image인 blob을 만듦

  <br>

  * age detection 수행

    ```python
    		# make predictions on the age and find the age bucket with
    		# the largest corresponding probability
    		ageNet.setInput(faceBlob)
    		preds = ageNet.forward()
    		i = preds[0].argmax()
    		age = AGE_BUCKETS[i]
    		ageConfidence = preds[0][i]
            
    		# display the predicted age to our terminal
    		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
    		print("[INFO] {}".format(text))
    		
            # draw the bounding box of the face along with the associated
    		# predicted age
    		y = startY - 10 if startY - 10 > 10 else startY + 10
    		cv2.rectangle(image, (startX, startY), (endX, endY),
    			(0, 0, 255), 2)
    		cv2.putText(image, text, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    # display the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    ```

    * 얼굴만 포함된 image인 blob을 사용하여 연령 버킷을 만들고 가장 확률이 높은 연령 버킷을 찾아 연령 예측
    * 얼굴만 포함된 image인 blob을 사용하여 연령 예측 (age bucket과 ageConfidence)
    * 얼굴 ROI의 좌표와 입력 데이터를 통해 예상 연령을 터미널에 표시하고 얼굴의 bounding box와 예상 연령의 이미지를 출력

<br>

### 06. OpenCV age detection results

* Run `python detect_age.py --image images/[your_image_name] --face face_detector --age age_detector`

1. 6살 사진 (2004년) - (8~12): 97.75%

   ![result01]()

2. 23살 사진 (2021년) - (8~12): 88.71%

   ![result02]()

3. 23살 사진 (2021년) - (15~20): 47.52%

   ![result03]()