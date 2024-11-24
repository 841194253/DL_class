# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 1. 读取图像
# image = cv2.imread('image/image5.png')  # 替换为你的图像路径
# if image is None:
#     raise FileNotFoundError("图像未找到，请检查路径！")
#
# # 转换为 HSV 色彩空间
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# # 2. 分割苹果区域（根据颜色范围）
# lower_red = np.array([0, 100, 100])  # 红色下限
# upper_red = np.array([10, 255, 255])  # 红色上限
# mask = cv2.inRange(hsv, lower_red, upper_red)
#
# # 清理噪声
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
# # 显示分割结果
# segmented = cv2.bitwise_and(image, image, mask=mask_cleaned)
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
# plt.title('Segmented Apple')
# plt.axis('off')
#
# # 3. 生成遮挡掩码（叶子部分检测）
# # 假设叶子为绿色区域
# lower_green = np.array([35, 50, 50])  # 绿色下限
# upper_green = np.array([85, 255, 255])  # 绿色上限
# leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
#
# # 合并叶子遮挡区域与苹果区域的掩码
# combined_mask = cv2.bitwise_or(mask_cleaned, leaf_mask)
#
# # 4. 修复遮挡区域
# restored = cv2.inpaint(image, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
#
# # 显示修复结果
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
# plt.title('Restored Image')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()

from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageFilter, ImageTk
import os
import tkinter.messagebox
import tkinter.ttk
import numpy as np
import cv2 as cv

sizex = 0
sizey = 0
quality = 100
path = ''
output_path = None
output_file = None
root = Tk()
root.geometry()
label_img = None

# 设置窗口标题
root.title('图片智能修复')


# 用于处理鼠标的OpenCV实用类
class Sketcher:

    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])
        # cv.imshow(self.windowname + ": mask", self.dests[1])

    # 鼠标处理的onMouse函数

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


# 载入图像
def loadimg():
    global path
    global sizex
    global sizey
    path = tkinter.filedialog.askopenfilename()
    lb.config(text=path)
    if path != '':
        try:
            img = Image.open(path)
            sizex = img.size[0]
            sizey = img.size[1]
            img = img.resize((400, 400), Image.ANTIALIAS)
            global img_origin
            img_origin = ImageTk.PhotoImage(img)
            global label_img
            label_img.configure(image=img_origin)
            label_img.pack()

        except OSError:
            tkinter.messagebox.showerror('错误', '图片格式错误，无法识别')


def inpaint(path):
    def function(img):
        try:

            # 创建一个原始图像的副本
            img_mask = img.copy()
            # 创建原始图像的黑色副本
            # Acts as a mask
            inpaintMask = np.zeros(img.shape[:2], np.uint8)

            # Create sketch using OpenCV Utility Class: Sketcher
            sketch = Sketcher('image', [img_mask, inpaintMask], lambda: ((255, 255, 255), 255))

            ch = cv.waitKey()

            if ch == ord('t'):
                # 使用Alexendra Telea提出的算法。快速行进法
                res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
                cv.imshow('Inpaint Output using FMM', res)
                cv.waitKey()
                cv.imwrite(path, res)

            if ch == ord('n'):
                # 使用Bertalmio, Marcelo, Andrea L. Bertozzi和Guillermo Sapiro提出的算法：Navier-Stokes, 流体动力学，以及图像和视频的绘制
                res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
                cv.imshow('Inpaint Output using NS Technique', res)
                cv.waitKey()
                cv.imwrite(path, res)

            if ch == ord('r'):
                img_mask[:] = img
                inpaintMask[:] = 0
                sketch.show()

            cv.destroyAllWindows()
        except ValueError as e:
            tkinter.messagebox.showerror('', repr(e))

    if path != '':
        try:
            img = Image.open(path)
            img1 = cv.imread(path, cv.IMREAD_COLOR)
            img1 = function(img1)


        except OSError:
            lb.config(text="您没有选择任何文件")
            tkinter.messagebox.showerror('错误', '图片格式错误，无法识别')

    else:
        tkinter.messagebox.showerror('错误', '未发现路径')


lb = Label(root, text='会在原路径保存图像')
lb.pack()

lb1 = Label(root, text='警告：会覆盖原图片', width=27, height=2, font=("Arial", 10), bg="red")
lb1.pack(side='top')

btn = Button(root, text="选择图片", command=loadimg)
btn.pack()

lb2 = Label(root, text='按下开始绘制想要修复的位置')
lb2.pack()

btn2 = Button(root, text="开始", command=lambda: inpaint(path))
btn2.pack()

lb3 = Label(root, text='绘制完成使用以下步骤')
lb3.pack()

lb4 = Label(root, text='t-使用FMM修复\nn-使用NS方法修复\nr-重新绘制区域')
lb4.pack()

label_img = tkinter.Label(root, text='原始图片')
label_img.pack()

root.mainloop()


