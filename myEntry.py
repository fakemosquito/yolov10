from ultralytics import YOLOv10
import os

def startTrain(data='data/data.yaml', epochs=500, batch=4, imgsz=640):
    model = YOLOv10('yolov10s.pt')
    model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz)

def startVal(trainOrder):
    modelPath = f'runs/detect/train{trainOrder}/weights/best.pt'
    model = YOLOv10(modelPath)
    model.val(data='data/data.yaml', batch=4)

def startPredict(trainOrder):
    model = YOLOv10(f'runs/detect/train{trainOrder}/weights/best.pt')
    results = model.predict(source='data/test/images', show=True)
    print(results)


if __name__ == '__main__':
    # startTrain(epochs=4, imgsz=128)

    startPredict(trainOrder='12')