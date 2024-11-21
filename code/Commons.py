import matplotlib.pyplot as plt
import numpy as np
import cv2


def plotImage(img, title, cmapType=None):
    # Display image
    if cmapType:
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def draw_Hough_lines(plate, lines):

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 300 * (-b))
        y1 = int(y0 + 300 * a)
        x2 = int(x0 - 300 * (-b))
        y2 = int(y0 - 300 * a)

        cv2.line(plate, (x1, y1), (x2, y2), (0, 0, 255), 2)