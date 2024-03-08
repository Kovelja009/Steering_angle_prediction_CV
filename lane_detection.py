import math

import OAK_D_api as oak
import cv2
import numpy as np
import math
import time

class State:
    # possible states: 0 <- initial polygon, 1 <- one lane detected, 2 <- two lanes detected
    def __init__(self, height, width):
        self.prev_angle = 0
        self.straight_line = int(width / 2), int(height), int(width / 2), int(height / 2)
        self.prev_line = self.straight_line
        self.prev_state = 0
        self.n_frames = 0
        self.horizon_height = int(height * 0.4)
        self.initial_shape = np.array(
            # left_down1, left_down2, left_up, right_up, right_down1, right_down2 <- points for region of interest
            [[(int(0), int(height)),(int(0), int(height * 0.8)), (int(width * 0.20), int(self.horizon_height)),
              (int(width * 0.8), int(self.horizon_height)), (int(width), int(height * 0.8)), (int(width), int(height))]])
        self.last_shape = self.initial_shape
        self.danger = False  # occurs when we don't find any lane


class AngleCalculator:
    def __init__(self, height, width, resize=1.0, draw_lines=False, decay=0.0):
        # self.oak_d = oak_d._
        self.resize = resize
        self.draw_lines = draw_lines
        self.height = int(height * resize)
        self.width = int(width * resize)
        self.EPSYLON = 0.000001
        self.decay = decay
        self.state = State(self.height, self.width)
        self.sharp_angles = (-40, 40)

    def get_angle(self, image):
        if self.resize != 1.0:
            image = self._resize(image)
        canny_image = self._canny(image)
        cropped_image = self._region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=150, minLineLength=30,
                                maxLineGap=90)
        left_line, right_line = self._average_slope_intercept(image, lines)
        central_line, angle = self._angle_calculator(image, left_line, right_line)
        if self.draw_lines:
            image = self._drawing_lines(image, lines, left_line, right_line, central_line, angle)

        return image, angle

    # draws hough lines, also average left and right lines, central line, angle or whether we have found any lines
    def _drawing_lines(self, image, lines, left_line, right_line, central_line, angle):
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line.reshape(4)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.putText(image, 'No lines detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image = self._display_lines(image, left_line, right_line, central_line)
            cv2.putText(image, f'Angle: {angle}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return image
    

    def _single_line_angle_calc(self, x1, x2, y2, isLeft=True):
        diff_x = abs(self.width / 2 - x1)
        if isLeft:
            x1_temp = int(x1 + diff_x)
            x2_temp = int(x2 + diff_x)
        else:
            x1_temp = int(x1 - diff_x)
            x2_temp = int(x2 - diff_x) 

        dx = int(x2_temp - self.width / 2)
        dy = int(y2 - self.height)
        theta = math.atan2(dy, dx)
        angle = math.degrees(theta) + 90

        return angle, x1_temp, x2_temp 

    def _angle_calculator(self, image, left_line, right_line):
        # if angle is 90 degrees than k is infinite, so we add EPSYLON to ensure that we aren't deviding by zero
        k_left = 0
        k_right = 0
        n_left = 0
        n_right = 0

        # if we haven't found any line but have searched whole ROI, just go straight
        if self.state.danger and left_line is None and right_line is None:
            self.state.prev_line = self.state.straight_line
            self.state.prev_angle = round(0.0, 2)
            return self.state.straight_line, self.state.prev_angle

        # if we haven't found any lines (maybe because of noise in image) but want to keep previous angle
        elif left_line is None and right_line is None:
            return self.state.prev_line, self.state.prev_angle
        x1, y1, x2, y2 = 0, 0, 0, 0
        ########################
        # ensuring safety via checking line availability: if we find
        # just one line, we will stick to that one
        ########################
        if (left_line is not None and right_line is None) or (right_line is not None and left_line is None):
            # only left line is found
            if left_line is not None:
                x1, y1, x2, y2 = left_line
                angle, x1_temp, x2_temp = self._single_line_angle_calc(x1, x2, y2, isLeft=True)
            else: # only right line is found
                x1, y1, x2, y2 = right_line
                angle, x1_temp, x2_temp = self._single_line_angle_calc(x1, x2, y2, isLeft=False)


            if angle < self.sharp_angles[0] or angle > self.sharp_angles[1]: # if angle is too big, we will stick to that line
                central_line = x1_temp, y1, x2_temp, y2
            else: # else we will go straight
                central_line = self.state.straight_line
                angle = 0.0
            self.state.prev_line = central_line
            curr_angle = round(angle, 2)

            # does smoothing for steering by not allowing for great fluctuation between frames
            adjusted_angle = round(self.state.prev_angle * self.decay + (1 - self.decay) * curr_angle, 2)
            self.state.prev_angle = adjusted_angle

            return central_line, adjusted_angle
        ########################
        # both lines are found
        ########################
        else:
            x1, y1, x2, y2 = left_line
            k_left = (y2 - y1) / (x2 - x1 + self.EPSYLON)
            n_left = y1 - (k_left * x1)

            x1, y1, x2, y2 = right_line
            k_right = (y2 - y1) / (x2 - x1 + self.EPSYLON)
            n_right = y1 - (k_right * x1)

            intersection_x = (n_right - n_left) / (k_left - k_right + self.EPSYLON)
            intersection_y = k_left * intersection_x + n_left
            central_line = int(self.width / 2), int(self.height), int(intersection_x), int(intersection_y)
        
            dx = int(intersection_x - self.width / 2)
            dy = int(intersection_y - self.height)
            theta = math.atan2(dy, dx)
            angle = math.degrees(theta) + 90

            self.state.prev_line = central_line
            curr_angle = round(angle, 2)

            # does smoothing for steering by not allowing for great fluctuation between frames
            adjusted_angle = round(self.state.prev_angle * self.decay + (1 - self.decay) * curr_angle, 2)
            self.state.prev_angle = adjusted_angle

            return central_line, adjusted_angle


    def _dynamic_ROI(self):
        if self.state.n_frames >= 15:  # should aproximately be about 1 sec (depends on fps)
            self.state.n_frames = 0
            self.state.prev_state = 0
            self.state.danger = False
            return self.state.initial_shape
        return self.state.last_shape

    # y = kx + n  [k = slope, n = intercept]
    def _average_slope_intercept(self, image, lines):
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
                    right_lines.append((k, n))               # [slope1, intercept1]
        #   axis = 0 means we are doing average column wise -> [slope2, intercept2]
        left_line = None
        right_line = None
        if len(left_lines) != 0:
            average_left = np.average(left_lines, axis=0)
            left_line = self._make_coordinates(average_left)
        if len(right_lines) != 0:
            average_right = np.average(right_lines, axis=0)
            right_line = self._make_coordinates(average_right)

        edge_case = False
        # setting state and shape
        ##################### no lines detected
        if len(left_lines) == 0 and len(right_lines) == 0:  # no lines detected in this frame
            edge_case = True
            self.state.n_frames += 1
            if self.state.n_frames >= 5:  # if we haven't find them for more than 5 frames then we change state
                self.state.last_shape = self.state.initial_shape
                self.state.prev_state = 0
                self.state.danger = True
                self.state.n_frames = 0
        ##################### detected one line
        if (len(left_lines) == 0 and len(right_lines) != 0) or (len(left_lines) != 0 and len(right_lines) == 0):
            edge_case = True
            self.state.prev_state = 1
            self.state.danger = False
            self.state.n_frames += 1
            if len(left_lines) != 0:
                self.state.last_shape = self._new_shape(left_line)
            else:
                self.state.last_shape = self._new_shape(right_line)

        if not edge_case:  # means we found both lines
            self.state.n_frames = 0
            self.state.danger = False
            self.state.prev_state = 2
            self.state.last_shape = self._new_shape(left_line, right_line)

        return left_line, right_line

    def _region_of_interest(self, image):
        polygon = self._dynamic_ROI()
        mask = np.zeros_like(image)
        for pol in polygon: # if we have more polygons, we need to fill them separately, because they might overlap
            temp_mask = np.zeros_like(image)
            cv2.fillPoly(temp_mask, [pol], 255)
            mask = cv2.bitwise_or(mask, temp_mask)
            # cv2.fillPoly(mask, [pol], 255)
        cv2.imshow("mask", mask)
        mask = cv2.bitwise_and(mask, image)
        cv2.imshow("ROI", mask)
        return mask

    def _canny(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # blur = cv2.GaussianBlur(gray_image, (5, 5), 0) # cv2.Canny already does blurring in
        canny = cv2.Canny(gray_image, 100, 150)
        cv2.imshow("canny", canny)
        return canny


    # sometimes there are some points that are out of bounds, so we need to check that and constrain them
    def _fast_check(self, width, height, x1, y1, x2, y2):
        x1 = int(max(0, min(x1, width)))
        x2 = int(max(0, min(x2, width)))
        y1 = int(max(0, min(y1, height)))
        y2 = int(max(0, min(y2, height)))


        return x1, y1, x2, y2

    def _display_lines(self, image, left_line, right_line, central_line):
        line_image = np.zeros_like(image)
        roi_image = np.zeros_like(image)
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            x1, y1, x2, y2 = self._fast_check(self.width, self.height, x1, y1, x2, y2)
            cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=10)
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            x1, y1, x2, y2 = self._fast_check(self.width, self.height, x1, y1, x2, y2)
            cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=10)

        x1, y1, x2, y2 = central_line
        cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 100), thickness=10)

        for pol in self.state.last_shape:
            temp_mask = np.zeros_like(image)
            cv2.fillPoly(temp_mask, pts=[pol], color=(0, 255, 0))
            roi_image = cv2.bitwise_or(roi_image, temp_mask)
        line_image = cv2.addWeighted(line_image, 1, roi_image, 0.2, 1)

        combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combo_image


    def _resize(self, image):
        dim = (self.width, self.height)
        output = cv2.resize(image, dim)
        return output


    def _new_shape(self, first_line, second_line=None):
        x1, y1, x2, y2 = first_line
        offset = int(self.width * 0.1)
        ld_x = max(0, x1 - offset)
        lu_x = max(0, x2 - offset)
        rd_x = min(self.width, x1 + offset)
        ru_x = min(self.width, x2 + offset)
        # if left down > right down, then we have to swap them
        if ld_x > rd_x:
            temp = ld_x
            ld_x = rd_x
            rd_x = temp
        if second_line is None:
            return np.array(
                [[(ld_x, y1), (lu_x, y2), (ru_x, y2), (rd_x, y1)]]
            )
        else:
            xx1, yy1, xx2, yy2 = second_line
            ld_xx = max(0, xx1 - offset)
            lu_xx = max(0, xx2 - offset)
            rd_xx = min(self.width, xx1 + offset)
            ru_xx = min(self.width, xx2 + offset)
            # if left down > right down, then we have to swap them
            if ld_xx > rd_xx:
                temp = ld_xx
                ld_xx = rd_xx
                rd_xx = temp
            return np.array(
                [[(ld_x, y1), (lu_x, y2), (ru_x, y2), (rd_x, y1)],
                 [(ld_xx, yy1), (lu_xx, yy2), (ru_xx, yy2), (rd_xx, yy1)]]
            )

    def _make_coordinates(self, param):
        k, n = param
        y1 = self.height  # starts at the bottom of the image
        y2 = int(self.state.horizon_height)  # finishes above y1

        # using formula y = kx + n
        x1 = int((y1 - n) / k)
        x2 = int((y2 - n) / k)
        return np.array([x1, y1, x2, y2])


# pipeline for interacting with OAK-D camera
def oak_d_pipeline():
    oak_d = oak.OAK_D()
    angle_calc = AngleCalculator(height=1080, width=1920, resize=0.4, decay=0.7, draw_lines=True)
    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        computed_frame, angle = angle_calc.get_angle(frame)
        cv2.imshow("Vidra_car", computed_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


# saving video as mp4
def save_video(frames, height, width, name):
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print("Video saved")

# NOTE: decay is not visible on the image just by looking at central line,
# but can be observed by looking at the angle in the top left corner

if __name__ == '__main__':
    should_save = False
    frames = []

    # pipeline for reading from classic video
    cap = cv2.VideoCapture("bfmc_2.mp4")
    
    # Get height and width of frames
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    angle_calc = AngleCalculator(height=height, width=width, resize=1, decay=0.2, draw_lines=True)
    height, width = angle_calc.height, angle_calc.width
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        computed_frame, angle = angle_calc.get_angle(frame)
        if should_save:
            frames.append(computed_frame)    
        cv2.imshow("Vidra_car", computed_frame)
        # time.sleep(0.05)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if should_save:
        print("Saving video...")
        save_video(frames=frames, height=height, width=width, name="output.mp4")
