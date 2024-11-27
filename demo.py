import os
from PIL import Image

# from ultralytics import YOLO # 疑似原代码的bug，不写清Yolov10就会默认使用Yolov8，导致报错
from ultralytics import YOLOv10 as YOLO # 正确的导入方式
import cv2  # 添加此导入用于处理颜色通道


model = YOLO("runs/detect/train5/weights/best.pt")
input_folder = "demo/input"
output_folder = "demo/output"


for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(input_folder, filename)
        try:
            results = model(file_path)

            # 确保有结果返回
            if results and len(results) > 0:
                result_image = results[0].plot()

                result_image_pil = Image.fromarray(result_image)  # 转换为 PIL.Image 格式

                result_image_path = os.path.join(output_folder, filename)  # 保留原文件名

                if os.path.exists(result_image_path):
                    base, ext = os.path.splitext(result_image_path)
                    result_image_path = f"{base}_processed{ext}"

                # 将处理过的结果图像转换为 RGB 格式
                # 很重要，没有这一步，结果图像的颜色通道会被转换为 BGR 格式，导致结果图像颜色不正确 ！！！
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
                result_image_pil = Image.fromarray(result_image_rgb)  # 转换为 PIL.Image 格式

                result_image_pil.save(result_image_path)

                print(f"Processed and saved: {result_image_path}")
            else:
                print(f"No results for: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
