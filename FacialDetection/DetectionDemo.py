import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")  #Using a baseline YOLO model, will swap for SAM in (very) near future

cap = cv2.VideoCapture(0)  #standard webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = results[0].plot()  #Box & label
    cv2.imshow("Face Detection (YOLOv8n-face)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"): #Code wont work without Hex 255 inclusion.
        break

cap.release()
cv2.destroyAllWindows()