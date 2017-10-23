import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import math
import cv2
import numpy as np
import glob


# Loading desired image in grayscale
img = cv2.imread('IMG_6720.JPG', 0)
img = cv2.resize(img, (600, 800))

# blurring together the boxes of the QR code to reduce noise 
# when drawing a minimum area rectangle around this region
blurred = cv2.GaussianBlur(img, (5, 5), 0)
ret,thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)

# Creating contours to for identifying regions of interest
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im2, contours, -1, (0,255,0), 3)
areas = [cv2.contourArea(contour) for contour in contours]
copy = areas[:]
copy.sort()
area_of_QR =  copy[len(areas) - 2]
contour_index_of_QR = areas.index(area_of_QR)
QR_contour = contours[contour_index_of_QR]

# Drawing a minimum area rectangle around just the desired area
# of the QR Code. We only want this region, since we only have 
# the real world dimentions of this square. 
rect = cv2.minAreaRect(QR_contour)
boxPoints = cv2.boxPoints(rect)
box = np.int0(boxPoints)
cv2.drawContours(im2,[box],0,(0,0,255),2)

# This intermediate output was used to identify that the rectangle
# was properly being draw around just the QR code, ensuring we are
# using the correct contours and area of just the QR. Also provides,
# sanity check for how well rectangle approximates the QR

cv2.imshow("QR Rectangle", im2)


### Calculating distance from the camera to QR Code ###

focal_length = 29 # focal length of iPhone 6 in mm
real_height = 88 # actual height of QR code in mm
image_height = 2448 # total height of image in pixels
obj_height = rect[1][1] # height of the QR code in the image in pixels
sensor_height = 8.467 # height of iPhone 6 camera sensor in mm

### Calculating the yaw angle of the camera ###
camera_yaw = rect[2]
print("Yaw angle of the iPhone position is: %f" % (camera_yaw))

# This is a physics formula found online at https://goo.gl/2i7oiq
# it relates all of the aforementioned variables and uses the concept
# of proportions to identify the distance from camera to QR code
dist_to_obj = (focal_length * real_height * image_height)/(obj_height * sensor_height)

# print box
sum_of_x_coords = 0
sum_of_y_coords = 0

# coordinates_array is a list of the corners of the box around the
# QR code.
coordinates_array = []
for elem in box:
	sum_of_x_coords += elem[0]
	sum_of_y_coords += elem[1]
	coordinates_array.append([elem[0], elem[1]])

# Indentifies center coordinates of QR code, so we can map this 
# to the coordinate (0,0,0)
box_center_x, box_center_y = sum_of_x_coords/4, sum_of_y_coords/4
camera_x, camera_y, camera_z = box_center_x, box_center_y, dist_to_obj

print("Coordinates of iPhone position are: X: %f, Y: %f, Z: %f" % (camera_x, camera_y, camera_z))


### Calculating roll angle of the camera ###

# Uses the coordinates of the box and calculates angle with respect to
# the plan formed by the 2D image. 
for coordinate in coordinates_array:
	if coordinate[0] > box_center_x and coordinate[1] > box_center_y:
		max_x, max_y = coordinate[0], coordinate[1]
coordinates_array.remove([max_x, max_y])
top_right_coordinate = (max_x, max_y)

min_x, min_y = 1000000000000000, 1000000000000000
for coordinate in coordinates_array:
	if coordinate[0] > box_center_x and coordinate[1] < box_center_y:
		min_x, min_y = coordinate[0], coordinate[1]
coordinates_array.remove([min_x, min_y])
bottom_right_coordinate = (min_x, min_y)
triangle_coordinates = [top_right_coordinate, bottom_right_coordinate]

# Use np arrays to easily find distances between coordinates and find
# sines and cosines
a = np.array([bottom_right_coordinate[0], bottom_right_coordinate[1]])
b = np.array([top_right_coordinate[0], top_right_coordinate[1]])

if (top_right_coordinate[0] > bottom_right_coordinate[0] and top_right_coordinate[1] > bottom_right_coordinate[1]):
	third_x = max(x for x, y in triangle_coordinates)
	third_y = min(y for x, y in triangle_coordinates)
	c = np.array([third_x, third_y])
	ca = c - a
	ab = b - a
	cosine_angle = np.dot(ca, ab) / (np.linalg.norm(ca) * np.linalg.norm(ab))
	camera_roll = np.arcsin(cosine_angle)
else:
	third_x = max(x for x, y in triangle_coordinates)
	third_y = max(y for x, y in triangle_coordinates)
	c = np.array([third_x, third_y])
	cb = c - b
	ab = b - a
	cosine_angle = np.dot(cb, ab) / (np.linalg.norm(cb) * np.linalg.norm(ab))
	radians_camera_roll = np.arcsin(cosine_angle)
	degrees_camera_roll = np.degrees(radians_camera_roll)

third_coordinate = (third_x, third_y)
triangle_coordinates.append(third_coordinate)

print("Roll angle of iPhone position is: %f" % (degrees_camera_roll))


# Drawing a rudimentary rectangle represeting where iPhone
# is roughly located relative to the QR Code. Dimensions of iPhone were modelled
# based on the actual dimensions of an iPhone 6. 
cv2.rectangle(img,(camera_x - 75, camera_y - 37),(camera_x + 75, camera_y + 37),(0,255,0),3)
cv2.imshow("iPhone Rectangle", img)

print("Please note, in output image the rectangle appears to be on top of the QR code. This is not the case. The iPhone is from the perspective of the image itself. Image is shown from the viewfinder of the iPhone.")
# print top_right_coordinate
# print bottom_right_coordinate
# print third_coordinate
# print triangle_coordinates
# print np.degrees(radians_camera_roll)

cv2.waitKey()
