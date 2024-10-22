import os
from PIL import Image
from ultralytics import YOLO

# Load the pre-trained YOLOv10-N model
model = YOLO("yolov10l.pt")

# 指定输入和输出文件夹路径
input_folder = "demo/input"
output_folder = "demo/output"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 只处理图片文件，如 .jpg 和 .png
    if filename.endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(input_folder, filename)  # 获取完整的输入文件路径
        results = model(file_path)  # 对每个图片文件执行推理

        # 获取带有标注的图像
        result_image = results[0].plot()  # 假设每次只处理一个图像并获取它的结果

        # 将处理过的结果图像转换为 PIL 图像
        result_image_pil = Image.fromarray(result_image)  # 转换为 PIL.Image 格式

        # 构建输出文件的完整路径，保持原文件名
        result_image_path = os.path.join(output_folder, filename)  # 保留原文件名

        # 保存结果图像
        result_image_pil.save(result_image_path)

        print(f"Processed and saved: {result_image_path}")  # 输出处理结果的路径
