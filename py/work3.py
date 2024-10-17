import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r'../images/image_start/work_3_start.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 100, 200)

# Plot the original and edge-detected images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Start Image')
axs[0].axis('off')

axs[1].imshow(edges, cmap='gray')
axs[1].set_title('End Image')
axs[1].axis('off')

plt.show()

output_path = '../images/image_end/work_3_end.jpg'
cv2.imwrite(output_path, edges)
print(f"Equalized image saved to '{output_path}'")