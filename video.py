import cv2 as cv

vid_capture = cv.VideoCapture(1)

while True:
    isTrue, frame = vid_capture.read(1)
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

vid_capture.release()
cv.destroyAllWindows()