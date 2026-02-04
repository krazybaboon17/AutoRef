import cv2
from flask import Flask, Response

app = Flask("AutoRef")
cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def gen():
    while True:
        r,f = cap.read()
        if not r: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for x,y,w,h in face.detectMultiScale(g,1.1,4):
            cv2.rectangle(f,(x,y),(x+w,y+h),(255,0,0),2)
        _,b = cv2.imencode(".jpg",f)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+b.tobytes()+b"\r\n"

@app.route("/video")
def v(): return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__": app.run()
