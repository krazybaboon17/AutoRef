import cv2
import numpy as np
from ultralytics import YOLO
import json
from flask import Flask, Response, render_template, request, jsonify

table_points = []

class BallTracker:
    def __init__(self):
        self.lowerOrange = np.array([5, 120, 100])
        self.upperOrange = np.array([25, 255, 255])
        
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

videoCapture = cv2.VideoCapture(0)
app = Flask(__name__)
yolo_model = YOLO('yolov8n.pt') 

def generateFrames():
    tracker = BallTracker()
    
    while True:
        isRet, currentFrame = videoCapture.read()
        if not isRet: continue
        
        results = yolo_model.predict(currentFrame, classes=[32], conf=0.25, verbose=False)
        yolo_ball = None
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy()
                yolo_ball = {"pos": (int((b[0]+b[2])/2), int((b[1]+b[3])/2)), "radius": int((b[2]-b[0])/2)}
                break 

        ballData = tracker.detectBall(currentFrame)
        
        if ballData:
            bp, br, bs = ballData["pos"], ballData["radius"], ballData["state"]
            cv2.circle(currentFrame, bp, br + 5, (0, 165, 255), 2)
            cv2.putText(currentFrame, "PINGPONG", (bp[0] + 10, bp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            if bs == "TRACKING":
                rx1, ry1 = max(0, bp[0] - tracker.roiSize), max(0, bp[1] - tracker.roiSize)
                rx2, ry2 = min(currentFrame.shape[1], bp[0] + tracker.roiSize), min(currentFrame.shape[0], bp[1] + tracker.roiSize)
                cv2.rectangle(currentFrame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        if yolo_ball:
            yp, yr = yolo_ball["pos"], yolo_ball["radius"]
            cv2.circle(currentFrame, yp, yr + 8, (255, 255, 0), 3)
            cv2.putText(currentFrame, "YOLO BALL", (yp[0] + 10, yp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        main_ball_pos = yolo_ball["pos"] if yolo_ball else (ballData["pos"] if ballData else None)
        
        if len(table_points) == 4:
            pts = np.array(table_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(currentFrame, [pts], True, (0, 255, 255), 2)
        
            min_x = min(p[0] for p in table_points)
            max_x = max(p[0] for p in table_points)
            min_y = min(p[1] for p in table_points)
            max_y = max(p[1] for p in table_points)
            cv2.rectangle(currentFrame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(currentFrame, "TABLE RANGE", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if main_ball_pos:
                bx = main_ball_pos[0]
                if bx < min_x or bx > max_x:
                    overlay = currentFrame.copy()
                    cv2.rectangle(overlay, (0, 0), (currentFrame.shape[1], currentFrame.shape[0]), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, currentFrame, 0.7, 0, currentFrame)
                    cv2.putText(currentFrame, "OUT!", (currentFrame.shape[1]//2 - 100, currentFrame.shape[0]//2), 
                                cv2.FONT_HERSHEY_DUPLEX, 4.0, (255, 255, 255), 3)

        success, encodedFrame = cv2.imencode(".jpg", currentFrame)
        if success: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedFrame.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def videoFeed(): return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_points', methods=['POST'])
def set_points():
    global table_points
    data = request.get_json()
    if 'points' in data:
        table_points = data['points']
        return jsonify({"status": "success", "points": table_points})
    return jsonify({"status": "error", "message": "No points provided"}), 400

@app.route('/get_points')
def get_points():
    return jsonify({"points": table_points})

if __name__ == "__main__": app.run(host='0.0.0.0', port=5001)
