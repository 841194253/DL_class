import cv2
import numpy as np

# 声明全局变量
img = None  # 原图
inpaintMask = None  # 掩码图像
prevPt = (-1, -1)  # 上一个鼠标点击点，用于绘制线条

# 鼠标回调函数，用于捕捉绘制区域
def onMouse(event, x, y, flags, param):
    global prevPt, img, inpaintMask
    # 如果左键释放或松开，不再记录鼠标位置
    if event == cv2.EVENT_LBUTTONUP or not (flags & cv2.EVENT_FLAG_LBUTTON):
        prevPt = (-1, -1)
    # 如果左键按下，记录当前点击位置
    elif event == cv2.EVENT_LBUTTONDOWN:
        prevPt = (x, y)
    # 如果鼠标在按下左键的情况下移动，绘制线条
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        pt = (x, y)
        # 如果 prevPt 是初始值(-1, -1)，则将当前鼠标位置设为 prevPt
        if prevPt[0] < 0:
            prevPt = pt
        # 在掩码图像上绘制线条，作为修复区域
        cv2.line(inpaintMask, prevPt, pt, (255), 5, 8, 0)
        # 同时在原图上绘制线条
        cv2.line(img, prevPt, pt, (255, 255, 255), 5, 8, 0)
        prevPt = pt
        cv2.imshow("image", img)  # 更新显示图像

def main():
    global img, inpaintMask
    # 加载图像
    filename = r"image\image5.png"  # 设置图像路径
    img = cv2.imread(filename, cv2.IMREAD_COLOR)  # 读取图像
    if img is None:  # 检查图像是否加载成功
        print("Couldn't open the image " + filename)
        return

    # 创建窗口并初始化掩码
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    inpaintMask = np.zeros(img.shape[:2], np.uint8)  # 初始化掩码为黑色，确保是8位单通道图像

    cv2.imshow("image", img)  # 显示原始图像
    cv2.setMouseCallback("image", onMouse)  # 设置鼠标回调函数

    while True:
        key = cv2.waitKey(0)  # 等待用户输入

        # 按下 ESC 键退出程序
        if key == 27:  # ESC键退出
            break

        # 按下 'r' 键恢复原图并清空掩码
        if key == ord('r'):  # 按 'r' 键恢复原图
            inpaintMask = np.zeros(img.shape[:2], np.uint8)  # 清空掩码
            img = cv2.imread(filename, cv2.IMREAD_COLOR)  # 重新加载原图
            cv2.imshow("image", img)

        # 按下 'i' 或空格键执行修复操作
        if key == ord('i') or key == ord(' '):  # 按 'i' 或 空格键执行修复
            # 使用 Telea 算法进行修复
            inpainted = cv2.inpaint(img, inpaintMask, 3, cv2.INPAINT_TELEA)
            cv2.imshow("inpainted image", inpainted)  # 显示修复后的图像

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

# 执行程序
if __name__ == "__main__":
    main()
