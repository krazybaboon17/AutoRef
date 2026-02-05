import cv2
from flask import Flask, Response, render_template
import numpy as np
import mediapipe as mp

lowerOrange = np.array([5, 150, 150])
upperOrange = np.array([15, 255, 255])
lowerWhite = np.array([0, 0, 200])
upperWhite = np.array([180, 30, 255])

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

app = Flask(__name__)
scaleFactor = 0.5

def detect_ball(frame, lowerColor, upperColor, colorBgr=(0,0,255), label='BALL'):
    blurredFrame = cv2.GaussianBlur(frame, (11,11),0)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvFrame, lowerColor, upperColor)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            ((xCenter, yCenter), radius) = cv2.minEnclosingCircle(c)
            contourArea = cv2.contourArea(c)
            circleArea = np.pi * (radius ** 2)
            roundness = contourArea / circleArea if circleArea > 0 else 0
            if roundness > 0.7:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w, y+h), colorBgr, 2)
                cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorBgr, 2)

def generate_frames():
    while True:
        isRet, currentFrame = videoCapture.read()
        if not isRet:
            continue

        resizedFrame = cv2.resize(currentFrame, None, fx=scaleFactor, fy=scaleFactor)
        grayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        detectedFaces = faceCascade.detectMultiScale(grayFrame, 1.1, 4)

        for (x, y, width, height) in detectedFaces:
            x, y, w, h = int(x/scaleFactor), int(y/scaleFactor), int(width/scaleFactor), int(height/scaleFactor)
            cv2.rectangle(currentFrame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(currentFrame, "FACE", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        detect_ball(currentFrame, lowerOrange, upperOrange, (0,0,255), 'BALL')
        detect_ball(currentFrame, lowerWhite, upperWhite, (255,255,255), 'BALL')

        rgbFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgbFrame)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(currentFrame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        success, encodedFrame = cv2.imencode(".jpg", currentFrame)
        if not success:
            continue
        frameBytes = encodedFrame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frameBytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
