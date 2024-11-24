# 13.给出采用形态学处理应用提取图像边界的实例一个。
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 读取图像
    src = cv2.imread("../image/image2.png")
    if src is None:
        print("无法读取图像文件！")
        return

    # 将 BGR 图像转换为 RGB 格式，以便在 plt 中显示
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    # 定义结构元素
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))

    # 执行腐蚀操作
    eroded = cv2.erode(src, element)
    eroded_rgb = cv2.cvtColor(eroded, cv2.COLOR_BGR2RGB)

    # 计算原始图像与腐蚀图像的差异
    diff = cv2.absdiff(src, eroded)

    # 将差异图像转换为 RGB 格式
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

    # 拼接图像（原图、腐蚀图、差异图）
    display = np.hstack((src_rgb, eroded_rgb, diff_rgb))

    # 显示拼接后的图像
    plt.figure(figsize=(12, 6))
    plt.imshow(display)
    plt.title('Original | Eroded | Difference')
    plt.axis('off')  # 关闭坐标轴
    plt.show()

if __name__ == "__main__":
    main()

