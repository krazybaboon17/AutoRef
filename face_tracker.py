import cv2
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)
while True:
    isRet, currentFrame = videoCapture.read()
    if not isRet:
        break
    grayImage = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
    detectedFaces = faceCascade.detectMultiScale(grayImage, 1.1, 4)
    for (x, y, width, height) in detectedFaces:
        cv2.rectangle(currentFrame, (x, y), (x+width, y+height), (255, 0, 0), 2)
    cv2.imshow('Face Tracker', currentFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videoCapture.release()
cv2.destroyAllWindows()