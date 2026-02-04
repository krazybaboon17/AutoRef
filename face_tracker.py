import cv2
from flask import Flask, Response

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)

app = Flask(__name__)

def generate_frames():
    while True:
        isRet, currentFrame = videoCapture.read()
        if not isRet:
            continue
        grayImage = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
        detectedFaces = faceCascade.detectMultiScale(grayImage, 1.1, 4)
        for (x, y, width, height) in detectedFaces:
            cv2.rectangle(currentFrame, (x, y), (x+width, y+height), (255, 0, 0), 2)

        success, encoded_image = cv2.imencode(".jpg", currentFrame)
        if not success:
            continue
        frame_bytes = encoded_image.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
