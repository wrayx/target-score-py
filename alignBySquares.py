import cv2
import sys
import numpy as np

# reading image
shotImgPath = sys.argv[1]
outputImgPath = sys.argv[2]

sourceImage = cv2.imread(sys.argv[1])
sourceImageCopy = sourceImage.copy()

sourceImage = cv2.GaussianBlur(sourceImage, (5, 5), 0)
# converting image into grayscale image
gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
# setting threshold of gray image
_, threshold = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY_INV)
# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

cv2.imshow('threshold image', threshold)

squareContours = []
squareContoursWithCorners = []

# list for storing names of shapes
for contour in contours:
    # contour = contours[-2]
    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    if len(approx) == 4:
        cv2.drawContours(sourceImage, [contour], 0, (255, 255, 255), 5)

        for app in approx:
            cv2.circle(sourceImage, app[0], 3, (0,0,0), 5)

        squareContoursWithCorners.append((contour, approx))
        squareContours.append(contour)
    
    
    if len(approx) >= 6:
        cv2.drawContours(sourceImage, [contour], 0, (255, 255, 0), 5)

squareContoursWithCorners.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
squareContours = sorted(squareContours, key=cv2.contourArea, reverse=True)

cv2.drawContours(sourceImage, squareContours, 1, (0, 0, 255), 7)


# displaying the image after drawing contours
cv2.imshow('shapes', sourceImage)
cv2.imwrite("output/shapes.jpg", sourceImage)



width, height = 1500, 1500
sourceCorners = np.array([squareContoursWithCorners[1][1][0][0], squareContoursWithCorners[1][1][1][0], squareContoursWithCorners[1][1][3][0], squareContoursWithCorners[1][1][2][0]])
targetCorners = np.array([(0,0),(0,height),(width,0),(width,height)])

print(sourceCorners)
print(targetCorners)

# Get matrix H that maps source_corners to target_corners
H, _ = cv2.findHomography(sourceCorners, targetCorners, cv2.RANSAC)
# Apply matrix H to source image.
transformed_image = cv2.warpPerspective(
    sourceImageCopy, H, (height, width))


cv2.imwrite(outputImgPath, transformed_image)
cv2.imshow(outputImgPath, transformed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
