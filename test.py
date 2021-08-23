from __future__ import print_function
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np
import imutils
import math


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # keypoint_original = cv2.drawKeypoints(
    #     im1Gray, keypoints1, None, color=(0, 255, 0), flags=0)
    # cv2.imshow('keypoint_original', keypoint_original)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    cv2.imshow('matches', imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def findContourArea(contours):
    areas = []
    for cnt in contours:
        cont_area = cv2.contourArea(cnt)
        areas.append(cont_area)
    
    return areas

def getContours(image):
    # imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholdedImage = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def getTargetImage(imageName):
    target = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    print('h={}w={}'.format(target.shape[0], target.shape[1]))
    height = target.shape[0]
    width = target.shape[1]
    adjust = 5
    half = int(width/5.5*0.2) - adjust
    endHalf = int(width-(width/5.5*0.2)) + adjust
    cropped_target = target[half:endHalf, half:endHalf]
    # cv2.imshow('cropped_target', cropped_target)
    return cropped_target

def findContourCentre(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    # cv2.drawContours(background, [c], -1, (0, 255, 0), 1.5)
    # cv2.circle(background, (cX, cY), 3, (0, 255, 0), -1)
    # cv2.putText(background, "center ({},{})".format(cX, cY), (cX + 20, cY),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return (cX, cY)

# def drawCentre()

def getImgDiff(original_img, new_img):
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(original_img, new_img, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours = getContours(thresh.copy())
    contours[:] = [x for x in contours if cv2.contourArea(x) <= 1000]
    
    (avgX, avgY) = (0, 0)
    for c in contours:
        (cX, cY) = findContourCentre(c)
        avgX = avgX + cX
        avgY = avgY + cY

    avgX = int(avgX / len(contours))
    avgY = int(avgY / len(contours))

    # cnts = imutils.grab_contours(cnts)
    # cv2.imshow("Thresh", thresh)
    return contours, (avgX, avgY)

def singleTargetSystem(contours):
    return 0

def fiveTargetSystem(contours):
    return 0

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

if __name__ == '__main__':

    # Read reference image
    refFilename = "target_ref.jpg"
    # refFilename = "ref_target_red.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "IMG_2748.jpg"
    # imFilename = "target_red_2.JPG"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpeg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)
    cv2.imshow('aligned', imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)

    imRegGray = cv2.cvtColor(imReg, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey', imRegGray)

    # detecting contours

    target = getTargetImage('test_target_14_original.png')
    newTarget = getTargetImage('test_target_14_new.png')

    shotContours, (shotX, shotY) = getImgDiff(target, newTarget)
    # cropped_test_target = test_target
    # cropped_test_target = test_target[230:230+1700, 230:230+1700]
    # cropped_test_target = cv2.blur(cropped_test_target, (3,3))
    background = np.zeros((target.shape[0], target.shape[1], 3))
    # canny_edged_img = cv2.Canny(cropped_test_target, 100, 250)
    ret, th_target = cv2.threshold(target, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresholded_test_target', cropped_th_test_target)
    contours = getContours(th_target)
    cv2.drawContours(background, contours, -1, (255, 255, 255), 1)
    cv2.drawContours(background, shotContours, -1, (255, 255, 0), 1)
    cv2.imshow('cropped_thresholded_test_target', background)

    # print("Areas found: ", findContourArea(contours))
    # print("Contours found: ", len(contours))

    sortedByCourtourArea = sorted(contours, key=cv2.contourArea)
    # print("Sorted Areas: ", findContourArea(sortedByCourtourArea))

    targetOuterContour = sortedByCourtourArea[len(sortedByCourtourArea)-1]

    (centreX, centreY) = findContourCentre(targetOuterContour)
    cv2.circle(background, (centreX, centreY), 3, (0, 255, 0), -1)
    cv2.putText(background, "center ({},{}) Area={}".format(centreX, centreY, cv2.contourArea(targetOuterContour)), (centreX + 20, centreY),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # show the image
    cv2.imshow("Image", background)
    # cv2.waitKey(0)
    
    cv2.circle(background, (shotX, shotY), 3, (0, 255, 0), -1)
    cv2.putText(background, "centre: ({}, {})".format(shotX, shotY), (shotX + 20, shotY),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # loop over the contours
    # for shotContour in shotContours:
    #     # draw the contour and center of the shape on the image
    #     # cv2.drawContours(background, [c], -1, (0, 255, 0), 1.5)
    #     cv2.putText(background, "Area={}".format(cv2.contourArea(shotContour)), (shotX + 20, shotY),
    #     #     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    
    # show the image
    # cv2.waitKey(0)

    distance = calculateDistance(centreX, centreY, shotX, shotY)
    print("shot distance: ", distance)

    circleDistance = cv2.pointPolygonTest(targetOuterContour, (centreX, centreY), True)/10
    print("circle distance: ", circleDistance)

    score = float("{:.1f}".format(11 - round((distance/circleDistance), 1)))
    print("final score: ", score)

    cv2.putText(background, "Score: {}".format(score), (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 7)

    cv2.imshow("combined_result", background)

    # test_target = cv2.imread('test_target.png', cv2.IMREAD_GRAYSCALE)
    # cropped_test_target = test_target[130:170+782, 130:170+782]
    # cropped_test_target = cv2.blur(imRegGray, (3,3))
    # background = np.zeros((imRegGray.shape[0], imRegGray.shape[1], 3))
    # # canny_edged_img = cv2.Canny(imRegGray, 100, 250)
    # ret, cropped_th_test_target = cv2.threshold(imRegGray, 150, 255, cv2.THRESH_BINARY_INV)
    # contours, hierarchy = cv2.findContours(
    #     cropped_th_test_target, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(background, contours, -1, (255, 255, 255), 2)
    # cv2.imshow('cropped_thresholded_test_target', background)

    # print("Areas found: ", findContourArea(contours))
    # print("Contours found: ", len(contours))

    # sortedByCourtourArea = sorted(contours, key=cv2.contourArea)
    # print("Sorted Areas: ", findContourArea(sortedByCourtourArea))
    # sortedByCourtourArea[:] = [x for x in sortedByCourtourArea if cv2.contourArea(x) <= 3900]
    # # contours = imutils.grab_contours(contours)
    # # loop over the contours
    # for c in sortedByCourtourArea:
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     if M["m00"] == 0:
    #         continue
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     # draw the contour and center of the shape on the image
    #     # cv2.drawContours(background, [c], -1, (0, 255, 0), 1.5)
    #     cv2.circle(background, (cX, cY), 3, (0, 255, 0), -1)
    #     cv2.putText(background, "center ({},{})".format(cX, cY), (cX + 20, cY),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #     # show the image
    #     cv2.imshow("Image", background)
    #     # cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
