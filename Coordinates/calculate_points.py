import cv2
import numpy as np
 
# white 배경화면 생성하기
img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
 
x1, x2 = 100, 400
y1, y2 = 100, 400
 
# 시작점(x1, y1)과 종료점(x2, y2)을 잇는 빨간 사각형을 그림
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
 
pt1 = 120, 50
pt2 = 300, 500
 
# 시작점(pt1)과 종료점(pt2)을 잇는 굵기가 2인 파란 선을 그림
cv2.line(img, pt1, pt2, (255,0,0), 2)
 
# x start, y start, x end, y end
imgRect = (x1, y1, x2-x1, y2-y1)
 
# pt1에서 pt2까지의 직선이 imgRect 사각형에 의해 절단되는 좌표점을 계산
# 선이 사각 영역 밖에 있으면 retval에 False를 반환
# 접점을 반환
retval, rpt1, rpt2 = cv2.clipLine(imgRect, pt1, pt2)    # 접점(선과 선이 만나는 곳)
 
if retval:	# 접점이 있다면
    # 중심이 rpt1인 초록색 반지름이 5인 점을 생성, thickness(선 두께)가 -1이면 안을 채우기
    cv2.circle(img, rpt1, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, rpt2, radius=5, color=(0, 255, 0), thickness=-1)
 
cv2.imshow('img', img)	# 이미지 보여주기
cv2.waitKey()
cv2.destroyAllWindows()
