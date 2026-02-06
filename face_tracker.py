import cv2
from flask import Flask, Response, render_template
import numpy as np
import mediapipe as mp

class BallTracker:
    def __init__(self):
        self.lowerOrange = np.array([5, 150, 100])
        self.upperOrange = np.array([15, 255, 255])
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05 
        
        self.ballPos = None
        self.ballRadius = 0
        self.state = "SEARCH"
        self.trackCount = 0
        self.lostCount = 0
        self.roiSize = 120

    def detectBall(self, frame):
        maskArea = np.zeros(frame.shape[:2], dtype=np.uint8)
        if self.state == "TRACKING" and self.ballPos is not None:
            x1, y1 = max(0, self.ballPos[0] - self.roiSize), max(0, self.ballPos[1] - self.roiSize)
            x2, y2 = min(frame.shape[1], self.ballPos[0] + self.roiSize), min(frame.shape[0], self.ballPos[1] + self.roiSize)
            maskArea[y1:y2, x1:x2] = 255
        else:
            maskArea.fill(255)

        fgMask = self.backSub.apply(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(hsv, self.lowerOrange, self.upperOrange)
        combinedMask = cv2.bitwise_and(cv2.bitwise_and(fgMask, colorMask), maskArea)
        combinedMask = cv2.morphologyEx(combinedMask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(combinedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        foundNow = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
                self.ballRadius = int(radius)
                foundNow = True
                self.lostCount = 0
                if self.state == "SEARCH":
                    self.trackCount += 1
                    if self.trackCount > 3: self.state = "TRACKING"

        prediction = self.kf.predict()
        self.ballPos = (int(prediction[0][0]), int(prediction[1][0]))
        
        if not foundNow:
            self.lostCount += 1
            if self.lostCount > 10:
                self.state, self.trackCount, self.ballPos = "SEARCH", 0, None
                return None
        
        
        return {"pos": self.ballPos, "radius": self.ballRadius, "state": self.state}

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
videoCapture = cv2.VideoCapture(0)
app = Flask(__name__)
scaleFactor = 0.4

def generateFrames():
    tracker = BallTracker()
    smoothBox, smoothFaceBox = None, None
    alpha, faceAlpha = 0.4, 0.3
    
    while True:
        isRet, currentFrame = videoCapture.read()
        if not isRet: continue
        
        resizedFrame = cv2.resize(currentFrame, None, fx=scaleFactor, fy=scaleFactor)
        grayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        sideFaces = list(profileCascade.detectMultiScale(grayFrame, 1.1, 4))
        for (x, y, w, h) in profileCascade.detectMultiScale(cv2.flip(grayFrame, 1), 1.1, 4):
            sideFaces.append([grayFrame.shape[1] - x - w, y, w, h])

        faceCandidate = sideFaces[0] if sideFaces else None
        if faceCandidate is None:
            faces = faceCascade.detectMultiScale(grayFrame, 1.1, 4)
            if len(faces) > 0: faceCandidate = list(faces[0])

        xf, yf, wf, hf = None, None, None, None
        if faceCandidate is not None:
            xfc, yfc, wfc, hfc = [int(v/scaleFactor) for v in faceCandidate]
            if smoothFaceBox is None: smoothFaceBox = [xfc, yfc, wfc, hfc]
            else:
                for i in range(4): smoothFaceBox[i] = int(faceAlpha * [xfc, yfc, wfc, hfc][i] + (1 - faceAlpha) * smoothFaceBox[i])
            xf, yf, wf, hf = smoothFaceBox
        else: smoothFaceBox = None

        rgbFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgbFrame)
        if results.pose_landmarks:
            h, w, _ = currentFrame.shape
            xCoords = [int(lm.x * w) for lm in results.pose_landmarks.landmark]
            yCoords = [int(lm.y * h) for lm in results.pose_landmarks.landmark]
            xMin, yMin, xMax, yMax = max(0, min(xCoords)-20), max(0, min(yCoords)-20), min(w, max(xCoords)+20), min(h, max(yCoords)+20)
            if yf is not None: yMin = min(yMin, yf)
            if smoothBox is None: smoothBox = [xMin, yMin, xMax, yMax]
            else:
                vals = [xMin, yMin, xMax, yMax]
                for i in range(4): smoothBox[i] = int(alpha * vals[i] + (1 - alpha) * smoothBox[i])
            cv2.rectangle(currentFrame, (smoothBox[0], smoothBox[1]), (smoothBox[2], smoothBox[3]), (0, 255, 0), 2)
            cv2.putText(currentFrame, "BODY", (smoothBox[0], smoothBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if xf is not None:
            cv2.rectangle(currentFrame, (xf, yf), (xf+wf, yf+hf), (255, 0, 0), 2)
            cv2.putText(currentFrame, "FACE", (xf, yf-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        ballData = tracker.detectBall(currentFrame)
        if ballData:
            bp, br, bs = ballData["pos"], ballData["radius"], ballData["state"]
            cv2.circle(currentFrame, bp, br + 5, (0, 165, 255), 2)
            cv2.putText(currentFrame, "BALL", (bp[0] + 10, bp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            if bs == "TRACKING":
                rx1, ry1 = max(0, bp[0] - tracker.roiSize), max(0, bp[1] - tracker.roiSize)
                rx2, ry2 = min(currentFrame.shape[1], bp[0] + tracker.roiSize), min(currentFrame.shape[0], bp[1] + tracker.roiSize)
                cv2.rectangle(currentFrame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        success, encodedFrame = cv2.imencode(".jpg", currentFrame)
        if success: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedFrame.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def videoFeed(): return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__": app.run(host='0.0.0.0', port=5001)
