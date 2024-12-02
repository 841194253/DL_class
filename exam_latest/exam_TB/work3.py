# 3.采用图像形态学获取图像中小米的粒数、每粒小米的最大投影面积或者外接最大正方形的长与宽
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image/image3.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 尝试不同的二值化方法
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # 调整阈值
# 或者使用自适应阈值
# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 不使用或减少形态学操作
# kernel = np.ones((2, 2), np.uint8)
# cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# 检测轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 使用 cv2.RETR_TREE 模式

# 计算小米颗粒的数量和面积
grain_count = len(contours)
areas = []
squares = []

for cnt in contours:
    # 计算面积
    area = cv2.contourArea(cnt)
    if area > 10:  # 忽略一些小面积噪声
        areas.append(area)
        # 获取最小外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        square_size = max(w, h)
        squares.append(square_size)
        # 绘制检测框（可选）
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Binary Image with Contours")
plt.imshow(binary, cmap='gray')

plt.show()

# 输出小米颗粒的数量和最大面积
print(f"小米数量: {grain_count}")
if areas:
    print(f"每粒小米的最大投影面积: {max(areas):.2f}")
    print(f"每粒小米的最大外接正方形边长: {max(squares):.2f}")
else:
    print("未检测到有效的小米颗粒")

# 代码逻辑分析
# 图像预处理：
# 图像被转换为灰度图，随后通过阈值分割得到二值图像。采用 cv2.THRESH_BINARY_INV 适合白色背景和黑色目标。
# 你还提供了自适应阈值作为备用方法，适用于光照不均的场景。
# 轮廓检测：
# 使用 cv2.findContours() 获取小米颗粒的轮廓。
# 轮廓过滤：通过面积过滤掉噪声（面积阈值 10 是合理的初步选择）。
# 面积和外接矩形计算：每个轮廓的面积通过 cv2.contourArea() 计算。
# 获取每粒小米的最小外接矩形（cv2.boundingRect），提取宽度和高度，并选择较大者作为外接正方形的边长。
# 结果可视化与输出：在原图上绘制绿色矩形框，标识检测的小米。输出小米数量、最大投影面积和最大外接正方形边长。