from __future__ import print_function
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import sys
import numpy as np
import imutils
import math

green = (69, 255, 83)
lightGreen = (204, 255, 204)
grey = (110, 110, 110)
red = (67, 57, 249)
DARKRED = (0, 0, 255)
DARKGREEN = (28, 168, 23)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUNDCOLOR = (32, 32, 32)

shotNum = sys.argv[1]

targetFileName = 'test_img_6/aligned_shot_{}.JPG'.format(int(shotNum)-1)
nextTargetFileName = 'test_img_6/aligned_shot_{}.JPG'.format(shotNum)

targetTemplateFileName = 'test_img_6/aligned_shot_0.JPG'

# targetFileName = 'test_img/IMG_ref.JPG'
# nextTargetFileName = 'test_img/aligned.JPG'

# targetTemplateFileName = 'test_img/IMG_ref.JPG'




def findContourArea(contours):
    areas = []
    for cnt in contours:
        cont_area = cv2.contourArea(cnt)
        areas.append(cont_area)

    return areas


def getContours(image):
    # imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholdedImage = cv2.threshold(
        image, 100, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def getTargetImageInGrey(imageName):
    target = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    print('h={}w={}'.format(target.shape[0], target.shape[1]))
    height = target.shape[0]
    width = target.shape[1]
    adjust = 5
    half = int(width/5.5*0.25) - adjust
    endHalf = int(width-(width/5.5*0.25)) + adjust
    cropped_target = target[half:endHalf, half:endHalf]
    # cv2.imshow('cropped_target', cropped_target)
    return cropped_target


def getTargetImage(imageName):
    target = cv2.imread(imageName, cv2.IMREAD_COLOR)
    print('h={}w={}'.format(target.shape[0], target.shape[1]))
    height = target.shape[0]
    width = target.shape[1]
    adjust = 5
    half = int(width/5.5*0.25) - adjust
    endHalf = int(width-(width/5.5*0.25)) + adjust
    cropped_target = target[half:endHalf, half:endHalf]
    # cv2.imshow('cropped_target', cropped_target)
    return cropped_target


def findContourCentre(contour):
    M = cv2.moments(contour)

    if (M["m00"] == 0):
        return (0, 0)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # if cv2.pointPolygonTest(contour, (cX, cY), True) < 400:
    #     return (0, 0)
    # draw the contour and center of the shape on the image
    # cv2.drawContours(background, [c], -1, (0, 255, 0), 1.5)
    # cv2.circle(background, (cX, cY), 3, (0, 255, 0), -1)
    # cv2.putText(background, "center ({},{})".format(cX, cY), (cX + 20, cY),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return (cX, cY)

# def drawCentre()


def getShotContour(original_img, new_img):

    background1 = np.full(
        (original_img.shape[0], original_img.shape[1], 3), BACKGROUNDCOLOR, np.uint8)
    background2 = np.full(
        (new_img.shape[0], new_img.shape[1], 3), BACKGROUNDCOLOR, np.uint8)

    original_img = cv2.medianBlur(original_img, 15)
    th_original = cv2.threshold(original_img, 150, 255, cv2.THRESH_BINARY)[1]
    contours = getContours(th_original)
    cv2.drawContours(background1, contours, -1, (255, 255, 255), 1)

    # cv2.imshow('median_blurred_original', original_img)
    # cv2.imshow('thresholded_original', th_original)

    new_img = cv2.medianBlur(new_img, 15)
    th_new = cv2.threshold(new_img, 150, 255, cv2.THRESH_BINARY)[1]
    contours = getContours(th_new)
    cv2.drawContours(background2, contours, -1, (255, 255, 255), 1)

    # cv2.imshow('median_blurred_new', new_img)
    # cv2.imshow('thresholded_new', th_new)

    diff = cv2.absdiff(cv2.GaussianBlur(th_original, (25, 25), 0),
                       cv2.GaussianBlur(th_new, (25, 25), 0))
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours = getContours(thresh.copy())    
    print('shot contours: ', findContourArea(contours))
    contours[:] = [x for x in contours if cv2.contourArea(x) < 10000 and cv2.contourArea(x) > 500]
    print('shot contours: ', findContourArea(contours))
    (avgX, avgY) = (0, 0)
    for c in contours:
        (cX, cY) = findContourCentre(c)

        avgX = avgX + cX
        avgY = avgY + cY

    avgX = int(avgX / len(contours))
    avgY = int(avgY / len(contours))

    numpy_horizontal = np.hstack((original_img, new_img, thresh))
    cv2.imshow("thresh", numpy_horizontal)

    # cv2.imshow("Thresh", thresh)
    return contours, (avgX, avgY)


def getTargetCentre(targetImgInGrey):
    (centreX, centreY) = (0, 0)
    gray_blurred = cv2.blur(targetImgInGrey, (1, 1))
    # cv2.imshow("Detected Circle blurs", gray_blurred)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=30,
                                        param2=50, minRadius=20, maxRadius=40)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            (centreX, centreY) = (a, b)

    return r, (centreX, centreY)


def singleTargetSystem(contours):
    return 0


def fiveTargetSystem(contours):
    return 0


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def drawTarget(background, centreX, centreY, circleRadiuses, color=BLACK):
    # Draw the circumference of the circle.
    # cv2.circle(background, (centreX, centreY), int(circleDistance/2), BLACK, 3)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(background, (centreX, centreY), 2, color, -1)

    for i in range(0, 11):
        radius = int(circleRadiuses[i])
        cv2.circle(background, (centreX, centreY), radius, color, 3)
        if i < 9:
            cv2.putText(background, str(i+1), (centreX-10, centreY -
                        radius+40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(background, str(i+1), (centreX-10, centreY +
                        radius-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(background, str(i+1), (centreX+radius -
                        40, centreY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(background, str(i+1), (centreX-radius +
                        20, centreY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

if __name__ == '__main__':

    target = getTargetImageInGrey(targetFileName)
    nextTarget = getTargetImageInGrey(nextTargetFileName)

    # detecting circles
    circleContours = []
    circleRadiuses = []
    targetTemplate = getTargetImageInGrey(targetTemplateFileName)
    # targetTemplate = cv2.GaussianBlur(target, (15, 15), 0)
    # setting threshold of gray image
    _, threshold = cv2.threshold(targetTemplate, 220, 255, cv2.THRESH_BINARY_INV)
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

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

        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        if len(approx) >= 6:
            cv2.drawContours(targetTemplate, [contour], 0, (255, 255, 0), 2)
            circleContours.append(contour)

    circleContours = sorted(circleContours, key=cv2.contourArea, reverse=True)

    for contour in circleContours:
        cv2.drawContours(targetTemplate, circleContours, 21, red, 3)
        (cX, cY) = findContourCentre(contour)
        cv2.circle(targetTemplate, (cX, cY), 3, red, -1)

    for i in range(0, 22, 2):
        (cX, cY) = findContourCentre(circleContours[i])
        outerContourRadius = cv2.pointPolygonTest(
            circleContours[i], (cX, cY), True)
        innerContourRadius = cv2.pointPolygonTest(
            circleContours[i+1], (cX, cY), True)
        circleRadiuses.append((outerContourRadius+innerContourRadius)/2)

    print(circleRadiuses)
    # displaying the image after drawing contours
    # cv2.imshow('shapes', targetTemplate)
    numpy_horizontal = np.hstack((target, targetTemplate))
    cv2.imshow("circle detection", numpy_horizontal)

    # ===================================================================

    innerCircleRadius, (centreX, centreY) = getTargetCentre(target.copy())
    shotContours, (shotX, shotY) = getShotContour(target, nextTarget)

    # ============================= Drawing =============================

    th_target = cv2.GaussianBlur(target, (5, 5), 0)
    th_next_target = cv2.GaussianBlur(nextTarget, (5, 5), 0)
    # background = np.zeros((target.shape[0], target.shape[1], 3))
    _, th_target = cv2.threshold(target, 150, 255, cv2.THRESH_BINARY)
    _, th_next_target = cv2.threshold(nextTarget, 150, 255, cv2.THRESH_BINARY)

    # Draw the circumference of the circle.
    cv2.circle(th_next_target, (centreX, centreY), innerCircleRadius, green, 4)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(th_next_target, (centreX, centreY), 2, green, -1)
    # cv2.imshow("Detected Circle ({} - {})".format(30, 40), nextTarget)
    cv2.putText(th_next_target, "CENTRE: ({}, {})".format(centreX, centreY), (centreX+40, centreY+50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, green, 4)

    numpy_horizontal = np.hstack((th_target, th_next_target))
    cv2.imshow("Threshold Images", numpy_horizontal)

    cv2.putText(nextTarget, "'CROPPED IMG'", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 5)
    cv2.putText(th_next_target, "'THRESHOLD IMG'", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 5)
    numpy_horizontal = np.hstack((nextTarget, th_next_target))
    cv2.imshow("Original Images", numpy_horizontal)

    background = np.full(
        (target.shape[0], target.shape[1], 3), BACKGROUNDCOLOR, np.uint8)

    contours = getContours(th_target)
    cv2.circle(background, (centreX, centreY), 5, grey, -1)
    cv2.drawContours(background, contours, -1, grey, 2)
    cv2.drawContours(background, shotContours, -1, green, -1)
    cv2.circle(background, (shotX, shotY), 15, red, -1)
    cv2.circle(background, (shotX, shotY), 3, BACKGROUNDCOLOR, -1)
    cv2.putText(background, "SHOT LOCATION: ({}, {})".format(shotX, shotY), (shotX + 50, shotY),
                cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)

    # cv2.putText(background, "Target Center ({},{})".format(centreX, centreY), (centreX + 20, centreY),
    #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ===================================================================

    distance = calculateDistance(centreX, centreY, shotX, shotY)
    circleDistance = innerCircleRadius * 2
    # score = float("{:.1f}".format(11 - round((distance/circleDistance), 1)))
    score = 0

    # testing
    for i in range(0, len(circleRadiuses)):
        if distance <= circleRadiuses[i] and distance > circleRadiuses[i+1]:
            score = i+1
            smallScore = 1-((distance - circleRadiuses[i+1])/(circleRadiuses[i]-circleRadiuses[i+1]))

            score = score + (smallScore)
            
            if score > 10:
                score = float("{:.1f}".format(10 + ((score-10)/2)))
            else:
                score = float("{:.1f}".format(score))

    # if score < 10:
    #     score = float("{:.1f}".format(11 - round((distance/circleDistance), 0)))

    print("shot distance: ", distance)
    print("circle distance: ", circleDistance)
    print("final score: ", score)

    # ============================= Drawing =============================

    cv2.putText(background, "DISTANCE: {:.2f}".format(distance), (shotX + 50, shotY + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)

    cv2.putText(background, "SCORE: {}".format(score), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, green, 6)

    # cv2.imwrite("test_img/out.png", background)
    numpy_horizontal = np.hstack(
        (getTargetImage(targetFileName), getTargetImage(nextTargetFileName), background))
    cv2.imshow("Results", numpy_horizontal)

    # ===================================================================

    # ============================= Drawing =============================

    displayTarget = np.full(
        (target.shape[0], target.shape[1], 3), BACKGROUNDCOLOR, np.uint8)

    drawTarget(displayTarget, centreX, centreY, circleRadiuses, grey)

    cv2.circle(displayTarget, (shotX, shotY), 20, red, -1)
    cv2.circle(displayTarget, (shotX, shotY), 4, BLACK, -1)

    cv2.putText(displayTarget, "LOCATION: ({},{})".format(shotX, shotY), (shotX + 50, shotY),
                cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3)

    cv2.putText(displayTarget, "SCORE: {}".format(score), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, green, 6)

    photo = change_brightness(getTargetImage(nextTargetFileName), -30)
    drawTarget(photo, centreX, centreY, circleRadiuses, red)
    cv2.circle(photo, (shotX, shotY), 8, green, -1)
    cv2.putText(photo, "LOCATION: ({},{})".format(shotX, shotY), (shotX + 50, shotY),
                cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3)
    cv2.putText(photo, "'MAPPING'", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 5)

    numpy_horizontal = np.hstack((photo, displayTarget))
    cv2.imshow("Display", numpy_horizontal)

    # ===================================================================

    cv2.waitKey(0)
    cv2.destroyAllWindows()
