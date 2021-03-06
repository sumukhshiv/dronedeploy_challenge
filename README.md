# dronedeploy_challenge

# Coding challange for DroneDeploy. 

## Problem Statement:

Option 1.

This zip file https://www.dropbox.com/s/afrfye7hvra7wvy/images.zip?dl=0 contains a number of images taken from different positions and orientations with an iPhone 6. Each image is the view of a pattern on a flat surface. The original pattern that was photographed is 8.8cm x 8.8cm and is included in the zip file. Write a Python program that will visualize (i.e. generate a graphic) where the camera was when each image was taken and how it was posed, relative to the pattern.

You can assume that the pattern is at 0,0,0 in some global coordinate system and are thus looking for the x, y, z and yaw, pitch, roll of the camera that took each image. Please submit a link to a Github repository contain the code for your solution. Readability and comments are taken into account too. You may use 3rd party libraries like OpenCV and Numpy. 

## General Idea:

I utilized several OpenCV modules, such as contouring and rectangular approximation to extract relevant information from the images. Using this information I performed a series of calculations to identify camera position and created a simple visualization identifying the camera position. For specifics, please look at the comments in `position_extractor.py`.