import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../image/image1.png')

# 转换为HSV颜色空间，便于提取绿色部分
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义绿色的HSV阈值范围
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# 创建绿色掩码
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 使用形态学操作去噪声
kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

# 计算绿色区域的像素数
green_pixels = cv2.countNonZero(green_mask)

# 估算绿色部分面积
# 计算图像的分辨率与实际宽度的比例
image_width_in_cm = 20  # 槽的实际宽度为20厘米
image_width_in_pixels = image.shape[1]
pixel_to_cm_ratio = image_width_in_cm / image_width_in_pixels

# 将绿色像素数转换为平方厘米面积
green_area_cm2 = green_pixels * (pixel_to_cm_ratio ** 2)

print(f"绿色植物部分的估算面积为: {green_area_cm2:.2f} 平方厘米")

# # 显示绿色提取效果（可选）
# cv2.imshow("Green Mask", green_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 使用matplotlib显示绿色掩码
plt.imshow(green_mask, cmap='gray')
plt.title("Green Mask")
plt.axis("off")  # 不显示坐标轴
plt.show()
