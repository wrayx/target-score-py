import cv2 as cv

# print image
target = cv.imread("target_red.jpeg")
(h,w,d) = target.shape
print("width = {}, height = {}, depth = {}".format(w,h,d))
cv.imshow("Target", target)

# print pixel
(B,G,R) = target[2000, 2000]
print("BRIGHTNESS = {}, GREEN = {}, RED = {}".format(w,h,d))

# 


# 

cv.waitKey(0)
cv.destroyAllWindows()