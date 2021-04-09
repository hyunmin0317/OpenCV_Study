import cv2
import numpy as np
 
img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
 
x, y = 256, 256
size = 200
 
for angle in range(0, 90, 10):
    rect = ((256, 256), (size, size), angle)	# 사각형
    box = cv2.boxPoints(rect).astype(np.int32)	# 사각형의 꼭지점 좌표 구하기
    r = np.random.randint(256)	# 랜덤한 숫자(0 ~ 255 사이)
    g = np.random.randint(256)
    b = np.random.randint(256)   
    cv2.polylines(img, [box], True, (r, g, b), 2)
    
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
