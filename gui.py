import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from ultralytics import YOLOv10 as YOLO  # 导入YOLO模型


def open_image():
    filepath = filedialog.askopenfilename(
        title="选择一张图片",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if filepath:
        load_image(filepath)

def load_image(filepath):
    original_image = Image.open(filepath)

    model = YOLO("runs/detect/train5/weights/best.pt")
    result = model(filepath)
    processed_image = result[0].plot()

    if isinstance(processed_image, np.ndarray):
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(processed_image)

    display_image(original_image, original_label)
    display_image(processed_image, processed_label)

def display_image(image, label):
    # 获取标签的尺寸
    label_width = label.winfo_width()
    label_height = label.winfo_height()

    # 设定图像的最大尺寸
    max_size = (label_width, label_height)
    image.thumbnail(max_size, Image.LANCZOS)

    # 转换为 PhotoImage 并显示
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.image = photo

def drag_and_drop(event):
    # 拖放事件处理
    filepath = event.data
    if os.path.isfile(filepath):
        load_image(filepath)

# 创建主窗口
root = TkinterDnD.Tk()
root.title("YOLOv10  ")
root.geometry("1600x800")

# 创建左侧原图显示区域
original_label = tk.Label(root, text="拖拽或点击选择图片", bg="lightgray")
original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# 创建右侧处理后图像显示区域
processed_label = tk.Label(root, text="处理后的图片", bg="lightgray")
processed_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# 添加按钮来选择图片
select_button = tk.Button(root, text="浏览选择图片", command=open_image)
select_button.pack(pady=10)

# 绑定拖放事件
original_label.drop_target_register(DND_FILES)
original_label.dnd_bind('<<Drop>>', drag_and_drop)

# 运行主循环
root.mainloop()
