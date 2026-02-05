import cv2
from flask import Flask, Response, render_template
import numpy as np

lowerOrange = np.array([5, 150, 150])
upperOrange = np.array([15, 255, 255])
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(1)

app = Flask(__name__)
scale = 0.5

def generate_frames():
    while True:
        isRet, currentFrame = videoCapture.read()
        if not isRet:
            continue
        resizedFrame = cv2.resize(currentFrame, None, fx=scale, fy=scale)
        grayImage = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        detectedFaces = faceCascade.detectMultiScale(grayImage, 1.1, 4)
        for (x, y, width, height) in detectedFaces:
            x, y, w, h = int(x/scale), int(y/scale), int(width/scale), int(height/scale)
            cv2.rectangle(currentFrame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(currentFrame, "FACE", (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        blurredImg = cv2.GaussianBlur(currentFrame, (11, 11), 0)
        hsvImage = cv2.cvtColor(blurredImg, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvImage, lowerOrange, upperOrange)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:                                           
            c = max(contours, key=cv2.contourArea)            
            if cv2.contourArea(c) > 50:
                (xCenter, yCenter), radius = cv2.minEnclosingCircle(c)
                contourArea = cv2.contourArea(c)
                circleArea = np.pi * (radius ** 2)
                roundness = contourArea / circleArea if circleArea > 0 else 0
                if roundness > 0.4:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(currentFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(currentFrame, 'BALL', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        success, encoded_image = cv2.imencode(".jpg", currentFrame)
        if not success:
            continue
        frame_bytes = encoded_image.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)