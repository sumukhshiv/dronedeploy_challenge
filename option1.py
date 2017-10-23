import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import math
import cv2
import numpy as np
import glob

img = cv2.imread('IMG_6725.JPG', 0)
img = cv2.resize(img, (600, 800))
blurred = cv2.GaussianBlur(img, (5, 5), 0)
ret,thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im2, contours, -1, (0,255,0), 3)
cv2.imshow("contours", im2)
areas = [cv2.contourArea(contour) for contour in contours]
copy = areas[:]
copy.sort()
area_of_QR =  copy[len(areas) - 2]
contour_index_of_QR = areas.index(area_of_QR)
QR_contour = contours[contour_index_of_QR]


rect = cv2.minAreaRect(QR_contour)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im2,[box],0,(0,0,255),2)
cv2.imshow("rectangle", im2)

# counter = 0
# max_index = 0
# max_area = 0
# while counter < len(areas):
# 	if areas[counter] > max_area:
# 		max_area = areas[counter]
# 		max_index = counter
# print "max_area: " + max_area
# print "max_index: " + max_index

print areas
print copy

cv2.waitKey()



# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# for fname in glob.glob('IMG_6719.JPG'):
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
#     if ret == True:
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         # Find the rotation and translation vectors.
#         ret,rvecs, tvecs, inliers = cv2.solvePnP(objp, corners2, mtx, dist)
#         # project 3D points to image plane
#         imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
#         img = draw(img,corners2,imgpts)
#         cv2.imshow('img',img)
#         k = cv2.waitKey(0) & 0xFF
#         if k == ord('s'):
#             cv2.imwrite(fname[:6]+'.png', img)


