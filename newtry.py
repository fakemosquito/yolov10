from ultralytics import YOLO

# Load the pre-trained YOLOv10-N model
model = YOLO("yolov10l.pt")
results = model("C:/Users\qifen\Desktop\jjj.jpg")
results[0].show()