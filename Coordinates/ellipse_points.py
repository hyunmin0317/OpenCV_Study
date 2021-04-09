import cv2
import numpy as np
 
img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
 
ptCenter = img.shape[0]//2, img.shape[1]//2
size = 200, 100
 
# 타원 그리기
cv2.ellipse(img, ptCenter, size, 0, 0, 360, (255, 0, 0))
 
# 중심점이 ptCenter, 크기가 size(x, y)이고, 각도 간격이 45도 마다 좌표 계산
pts1 = cv2.ellipse2Poly(ptCenter, size,  0, 0, 360, delta=45)   # 원을 45도 각도마다 점을 만들기
print(pts1) # 점의 좌표를 출력
 
# 타원 그리기
cv2.ellipse(img, ptCenter, size, 45, 0, 360, (255, 0, 0))
pts2 = cv2.ellipse2Poly(ptCenter, size, 45, 0, 360, delta=45)	# 원을 45도 각도마다 점을 만들기
 
# 다각형 그리기
cv2.polylines(img, [pts1, pts2], isClosed=True, color=(0, 0, 255))
 
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
