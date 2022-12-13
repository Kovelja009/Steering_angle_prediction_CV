import OAK_D_api as oak
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def _region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    triangle = np.array([[(int(width * 0.2), height), (int(width * 0.5), 400), (int(width * 0.8), height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(mask, image)
    return mask


def _make_coordinates(image, param):
    a, b = param
    y1 = image.shape[0]  # starts at the bottom of the image
    y2 = int(y1 / 2)  # finishes above y1

    # using formula y = ax + b
    x1 = int((y1 - b) / a)
    x2 = int((y2 - b) / a)
    return np.array([x1, y1, x2, y2])


def _average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # first degree polynomial
            res = np.polyfit((x1, x2), (y1, y2), 1)
            a = res[0]
            b = res[1]
            if a < 0:
                left_lines.append((a, b))
            else:
                right_lines.append((a, b))  # [slope1, intercept1]
        #         axis = 0 means we are doing average column wise -> [slope2, intercept2]
    left_line = None
    right_line = None
    if len(left_lines) != 0:
        average_left = np.average(left_lines, axis=0)
        left_line = _make_coordinates(image, average_left)
    if len(right_lines) != 0:
        average_right = np.average(right_lines, axis=0)
        right_line = _make_coordinates(image, average_right)
    return left_line, right_line


def _display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None and len(lines) != 0:
        for line in lines:
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)
                cv2.line(line_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=10)
    return line_image


def get_angle(image):
    canny_image = _canny(image)
    cropped_image = _region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=20)
    average_lines = _average_slope_intercept(image, lines)
    line_image = _display_lines(image, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return combo_image, 50


if __name__ == '__main__':
    oak_d = oak.OAK_D()
    img = plt.imread('eva02.png')
    plt.imshow(img)
    plt.show()
    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        computed_frame, angle = get_angle(frame)
        cv2.imshow("VidraCar", computed_frame)
        if cv2.waitKey(1) == ord('q'):
            break
