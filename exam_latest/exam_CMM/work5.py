# 5.还原图中被部分遮挡的苹果
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
from PIL import Image, ImageTk
import os
import tkinter.messagebox
import numpy as np
import cv2 as cv

path = ''
original_img = None  # 原始图片
current_img = None  # 当前绘制的图片
inpaintMask = None  # 修复掩膜
sketch = None  # 绘制工具
original_format = ''  # 保存原始图片格式

root = Tk()
root.title('图片智能修复')
root.geometry('900x700')

# 显示文件路径
lb_path = Label(root, text='请选择图片文件', wraplength=400)
lb_path.pack()

# 创建用于显示原始图片和修改后图片的框
frame = Frame(root)
frame.pack()

# 原始图片框
original_frame = Frame(frame, width=400, height=400, bg='lightgrey')
original_frame.grid(row=0, column=0, padx=5, pady=5)
original_label = Label(original_frame, text='原始图片')
original_label.pack(fill=BOTH, expand=True)

# 修改后图片框
modified_frame = Frame(frame, width=400, height=400, bg='lightgrey')
modified_frame.grid(row=0, column=1, padx=5, pady=5)
modified_label = Label(modified_frame, text='修改后的图片')
modified_label.pack(fill=BOTH, expand=True)

# 保存文件名输入框
save_name_var = StringVar(value='修复结果')
save_name_label = Label(root, text="保存文件名：")
save_name_label.pack()
save_name_entry = Entry(root, textvariable=save_name_var, width=30)
save_name_entry.pack()

# 提示信息
lb_info = Label(root, text="按下 't' 使用 FMM 修复, 按下 'n' 使用 NS 修复, 按下 'r' 重新绘制", wraplength=500)
lb_info.pack()

class Sketcher:
    """用于绘制修复区域的工具"""
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

# 设置按钮功能
def load_image():
    global path, original_img, current_img, inpaintMask, sketch, original_format
    path = tkinter.filedialog.askopenfilename(title='选择图片', filetypes=[('图像文件', '*.jpg *.png *.bmp')])
    if not path:
        tkinter.messagebox.showwarning('警告', '未选择文件')
        return

    lb_path.config(text=f"已选择文件：{path}")
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        tkinter.messagebox.showerror('错误', '无法加载图片')
        return

    # 获取原始图片格式
    original_format = os.path.splitext(path)[1].lower()

    # 初始化图像和掩膜
    original_img = img.copy()
    current_img = img.copy()
    inpaintMask = np.zeros(img.shape[:2], np.uint8)

    # 更新原始图片和修改后图片显示
    update_original_preview(img)
    update_modified_preview(img)

def update_original_preview(img):
    """更新原始图片的预览"""
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    original_label.configure(image=img_tk)
    original_label.image = img_tk

def update_modified_preview(img):
    """更新修改后图片的预览"""
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    modified_label.configure(image=img_tk)
    modified_label.image = img_tk

def start_edit():
    global sketch
    if original_img is None:
        tkinter.messagebox.showwarning('警告', '请先加载图片')
        return

    def color_func():
        return ((255, 255, 255), 255)  # 白色画笔

    sketch = Sketcher('image', [current_img, inpaintMask], color_func)
    cv.imshow('image', current_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    update_modified_preview(current_img)  # 更新修改后的预览

def apply_inpaint(method='FMM'):
    """应用修复算法"""
    global current_img
    if original_img is None:
        tkinter.messagebox.showwarning('警告', '请先加载图片')
        return

    flags = cv.INPAINT_TELEA if method == 'FMM' else cv.INPAINT_NS
    current_img = cv.inpaint(original_img, inpaintMask, inpaintRadius=3, flags=flags)
    update_modified_preview(current_img)

def save_image():
    """保存最终结果"""
    global current_img, path, original_format
    if current_img is None:
        tkinter.messagebox.showwarning('警告', '没有可保存的图片')
        return

    save_name = save_name_var.get().strip()
    if not save_name:
        tkinter.messagebox.showwarning('警告', '保存文件名不能为空')
        return

    save_path = os.path.join(os.path.dirname(path), save_name + original_format)
    if original_format not in ['.jpg', '.png', '.bmp']:
        save_path += '.jpg'  # 默认保存为 JPG

    cv.imwrite(save_path, current_img)
    tkinter.messagebox.showinfo('成功', f'图片已保存到：\n{save_path}')

# 按键事件处理
def key_event_handler(event):
    """处理键盘事件"""
    if event.char == 't':
        apply_inpaint(method='FMM')
    elif event.char == 'n':
        apply_inpaint(method='NS')
    elif event.char == 'r':
        start_edit()

# 绑定按键事件
root.bind("<Key>", key_event_handler)

# 按钮布局
btn_load = Button(root, text="选择图片", command=load_image)
btn_load.pack()

btn_save = Button(root, text="保存图片", command=save_image)
btn_save.pack()

# Tkinter 主循环
root.mainloop()

# 这段代码实现了一个基于 Tkinter 的 GUI 程序，用于图像修复操作，结合了 OpenCV 的绘制功能和两种修复算法（FMM 和 NS）





