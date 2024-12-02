import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取输入图像
image = cv2.imread('image/image1.png')

# 将图像从BGR转换为HSV色彩空间
# HSV空间更适合从图像中分离绿色区域
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色的HSV范围
# 这两个数组分别表示绿色的下限和上限阈值
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# 创建绿色区域的掩码
# 掩码中的绿色部分为白色（255），其他部分为黑色（0）
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 应用形态学操作去除小噪声并优化掩码
# 使用3x3的矩阵进行卷积
kernel = np.ones((3, 3), np.uint8)
# 开操作：去除小的杂散噪声
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
# 闭操作：填补区域中的空隙
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

# 统计绿色区域的像素数量
green_pixel_count = cv2.countNonZero(green_mask)

# 估算绿色区域的实际面积
# 假设已知植物槽的实际宽度为20厘米
tray_width_cm = 20
# 获取图像宽度（以像素为单位）
image_width_pixels = image.shape[1]
# 计算图像中每个像素与实际厘米之间的比例
pixel_to_cm_ratio = tray_width_cm / image_width_pixels
# 计算绿色区域的实际面积，单位为平方厘米
green_area_cm2 = green_pixel_count * (pixel_to_cm_ratio ** 2)

# 输出估算的绿色区域面积
print(f"Estimated green area: {green_area_cm2:.2f} cm²")

# 使用matplotlib展示绿色区域掩码
plt.imshow(green_mask, cmap='gray')
plt.title("Extracted Green Area")
plt.axis("off")
plt.show()

# 代码逻辑分析
# 1. 读取图像并进行颜色空间转换：
#    图像被转换为HSV色彩空间，这种空间非常适合提取和分割特定颜色，特别是绿色区域。
# 2. 定义绿色阈值：
#    `lower_green` 和 `upper_green` 用来确定绿色的HSV范围，从而提取该范围内的像素。
# 3. 掩码创建与形态学优化：
#    通过 `cv2.inRange()` 方法生成绿色掩码，并利用形态学操作（开操作和闭操作）去除噪声并优化掩码质量。
# 4. 面积计算：
#    通过槽宽（20cm）和图像宽度的像素数来计算像素与实际距离的比例，再将绿色像素数转换为实际面积（平方厘米）。
# 5. 可视化展示：
#    使用 `matplotlib` 展示提取出的绿色区域掩码，便于验证分割效果。
