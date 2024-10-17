import cv2
import matplotlib.pyplot as plt

image_path = r'../images/image_start/work_3_start.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Image not loaded from path '{image_path}'. Please check the file path.")
else:
    # 应用高斯模糊以减少噪声 高斯模糊（5,5）应用核大小为5x5和sigma为1.4的平滑处理。
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # 应用Canny边缘检测 较低的阈值设置为100，较高的阈值设置为200。梯度大于200的像素被认为是强边，而那些梯度在100到200之间的像素如果连接到强边，则被认为是弱边。
    edges = cv2.Canny(blurred_image, 100, 200)

    # 绘制原始图像和边缘检测图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Start Image')
    axs[0].axis('off')

    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title('End Image')
    axs[1].axis('off')

    plt.show()

    output_path = r'../images/image_end/work_3_end.jpg'
    cv2.imwrite(output_path, edges)
    print(f"image saved to '{output_path}'")