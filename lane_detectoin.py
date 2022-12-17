import math

import OAK_D_api as oak
import cv2
import numpy as np
import math

EPSYLON = 0.000001


def _display_lines(image, left_line, right_line, central_line):
    line_image = np.zeros_like(image)
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=10)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=10)

    x1, y1, x2, y2 = central_line
    cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 100), thickness=10)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo_image


def _resize(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    dsize = (width, height)
    # resize image
    output = cv2.resize(image, dsize)
    return output


def _make_coordinates(image, param):
    k, n = param
    y1 = image.shape[0]  # starts at the bottom of the image
    y2 = int(y1 / 2)  # finishes above y1

    # using formula y = kx + n
    x1 = int((y1 - n) / k)
    x2 = int((y2 - n) / k)
    return np.array([x1, y1, x2, y2])


def _canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(gray_image, 70, 150)
    cv2.imshow("canny", canny)
    return canny

def _region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    # triangle = np.array([[(int(width * 0.2), height), (int(width * 0.45), int(height * 0.4)),
    #                       (int(width * 0.55), int(height * 0.4)), (int(width * 0.8), height)]])
    triangle = np.array([[(int(width * 0.15), height), (int(width * 0.35), int(height * 0.4)),
                          (int(width * 0.6), int(height * 0.4)), (int(width * 0.85), height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(mask, image)
    cv2.imshow("ROI", mask)
    return mask


# y = kx + n  [k = slope, n = intercept]
def _average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # first degree polynomial
            res = np.polyfit((x1, x2), (y1, y2), 1)
            k = res[0]
            n = res[1]
            if k < 0:
                left_lines.append((k, n))
            else:
                right_lines.append((k, n))  # [slope1, intercept1]
    #   axis = 0 means we are doing average column wise -> [slope2, intercept2]
    left_line = None
    right_line = None
    if len(left_lines) != 0:
        average_left = np.average(left_lines, axis=0)
        left_line = _make_coordinates(image, average_left)
    if len(right_lines) != 0:
        average_right = np.average(right_lines, axis=0)
        right_line = _make_coordinates(image, average_right)
    return left_line, right_line


def _angle_calculator(image, left_line, right_line):
    # if angle is 90 degrees than k is infinite, so we add EPSYLON to ensure that we aren't deviding with zero
    height = image.shape[0]
    width = image.shape[1]

    k_left = 0
    k_right = 0
    n_left = 0
    n_right = 0
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        k_left = (y2 - y1) / (x2 - x1 + EPSYLON)
        n_left = y1 - (k_left * x1)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        k_right = (y2 - y1) / (x2 - x1 + EPSYLON)
        n_right = y1 - (k_right * x1)
    intersection_x = (n_right - n_left) / (k_left - k_right + EPSYLON)
    intersection_y = k_left * intersection_x + n_left

    # s = (intersection_x - width/2) - (intersection_y - height)
    central_line = int(width / 2), int(height), int(intersection_x), int(intersection_y)
    # steering_angle = 90 - (np.arctan(1/s) * (180/np.pi))
    dx = int(intersection_x - width / 2)
    dy = int(intersection_y - height)
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta) + 90

    return central_line, round(angle, 2)


def get_angle(image, resize=None, draw_lines=False):
    if resize is not None:
        image = _resize(image, resize)
    canny_image = _canny(image)
    cropped_image = _region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=30, maxLineGap=200)
    left_line, right_line = _average_slope_intercept(image, lines)
    central_line, angle = _angle_calculator(image, left_line, right_line)
    if draw_lines:
        image = _display_lines(image, left_line, right_line, central_line)

    return image, angle


if __name__ == '__main__':
    # oak_d = oak.OAK_D()
    # img = plt.imread('eva02.png')
    # plt.imshow(img)
    # plt.show()
    # while True:
    #     frame = oak_d.get_color_frame(show_fps=True)
    #     computed_frame, angle = get_angle(frame, 0.5)
    #     cv2.imshow("VidraCar", computed_frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    cap = cv2.VideoCapture("test_video5.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        computed_frame, angle = get_angle(frame, resize=0.4, draw_lines=True)

        cv2.imshow("Vidra_car", computed_frame)
        print(f'Angle: {angle}')
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
