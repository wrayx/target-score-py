import sys
import cv2
import numpy as np
import math

MAX_FEATURES = 800
GOOD_MATCH_PERCENT = 0.25

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
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("output/matches.jpg", imMatches)
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


if __name__ == '__main__':

    # Read reference image
    refFilename = sys.argv[1]
    # refFilename = "ref_target_red.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = sys.argv[2]
    # imFilename = "target_red_2.JPG"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = sys.argv[3]
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)
    cv2.imshow('aligned', imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)

    # imRegGray = cv2.cvtColor(imReg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grey', imReg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()