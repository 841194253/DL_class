import cv2
import matplotlib.pyplot as plt
import os

image_path = r'../images/image_start/work_2_start.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Image not loaded from path '{image_path}'. Please check the file path.")
else:
    # 应用直方图均衡
    equalized_image = cv2.equalizeHist(image)

    # 绘制原图和均衡后的图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Start Image')
    axs[0].axis('off')

    axs[1].imshow(equalized_image, cmap='gray')
    axs[1].set_title('End Image')
    axs[1].axis('off')

    plt.show()
    output_path = '../images/image_end/work_2_end.jpg'
    cv2.imwrite(output_path, equalized_image)
    print(f"Equalized image saved to '{output_path}'")