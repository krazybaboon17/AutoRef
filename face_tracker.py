import cv2
from flask import Flask, Response, render_template
import numpy as np
import mediapipe as mp



faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

videoCapture = cv2.VideoCapture(0)





app = Flask(__name__)
scaleFactor = 0.2



def generateFrames():
    smoothBox = None
    smoothFaceBox = None
    alpha = 0.4
    faceAlpha = 0.3
    while True:
        isRet, currentFrame = videoCapture.read()
        if not isRet:
            continue

        resizedFrame = cv2.resize(currentFrame, None, fx=scaleFactor, fy=scaleFactor)
        grayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        sideFaces = list(profileCascade.detectMultiScale(grayFrame, 1.1, 4))
        for (x, y, w, h) in profileCascade.detectMultiScale(cv2.flip(grayFrame, 1), 1.1, 4):
            sideFaces.append([grayFrame.shape[1] - x - w, y, w, h])

        faceCandidate = None
        if sideFaces:
            faceCandidate = sideFaces[0]
        else:
            faces = faceCascade.detectMultiScale(grayFrame, 1.1, 4)
            if len(faces) > 0:
                faceCandidate = list(faces[0])

        xf, yf, wf, hf = (None, None, None, None)
        if faceCandidate is not None:
            xfc, yfc, wfc, hfc = faceCandidate
            xfc, yfc, wfc, hfc = int(xfc/scaleFactor), int(yfc/scaleFactor), int(wfc/scaleFactor), int(hfc/scaleFactor)
            
            if smoothFaceBox is None:
                smoothFaceBox = [xfc, yfc, wfc, hfc]
            else:
                smoothFaceBox[0] = int(faceAlpha * xfc + (1 - faceAlpha) * smoothFaceBox[0])
                smoothFaceBox[1] = int(faceAlpha * yfc + (1 - faceAlpha) * smoothFaceBox[1])
                smoothFaceBox[2] = int(faceAlpha * wfc + (1 - faceAlpha) * smoothFaceBox[2])
                smoothFaceBox[3] = int(faceAlpha * hfc + (1 - faceAlpha) * smoothFaceBox[3])
            
            xf, yf, wf, hf = smoothFaceBox
        else:
            smoothFaceBox = None

        
        rgbFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgbFrame)

        if results.pose_landmarks:
            h, w, _ = currentFrame.shape
            xMin, yMin = w, h
            xMax, yMax = 0, 0
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                xMin, yMin = min(xMin, cx), min(yMin, cy)
                xMax, yMax = max(xMax, cx), max(yMax, cy)
            
            
            xMin, yMin = max(0, xMin - 20), max(0, yMin - 20)
            xMax, yMax = min(w, xMax + 20), min(h, yMax + 20)
            
            
            if yf is not None:
                yMin = min(yMin, yf)

            if smoothBox is None:
                smoothBox = [xMin, yMin, xMax, yMax]
            else:
                smoothBox[0] = int(alpha * xMin + (1 - alpha) * smoothBox[0])
                smoothBox[1] = int(alpha * yMin + (1 - alpha) * smoothBox[1])
                smoothBox[2] = int(alpha * xMax + (1 - alpha) * smoothBox[2])
                smoothBox[3] = int(alpha * yMax + (1 - alpha) * smoothBox[3]) 
            
            xb, yb, xe, ye = smoothBox
            cv2.rectangle(currentFrame, (xb, yb), (xe, ye), (0, 255, 0), 2)
            cv2.putText(currentFrame, "BODY", (xb, yb-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        if xf is not None:
            cv2.rectangle(currentFrame, (xf, yf), (xf+wf, yf+hf), (255, 0, 0), 2)
            cv2.putText(currentFrame, "FACE", (xf, yf-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)




        success, encodedFrame = cv2.imencode(".jpg", currentFrame)
        if not success:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedFrame.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def videoFeed():
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
