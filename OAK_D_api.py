import cv2
import depthai as dai
import time


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

        self._coordinates = (30, 50)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._color = (0, 0, 255)
        self._thickness = 2

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

    def show_fps(self, frame, fps):
        return cv2.putText(frame, fps.__str__(), self._coordinates, self._font, self._font_scale, self._color, self._thickness, cv2.LINE_AA)


class OAK_D:
    def __init__(self, fps=24, width=720, height=480):
        # Create pipeline
        self._pipeline = dai.Pipeline()

        # Define source and output
        self._camRgb = self._pipeline.create(dai.node.ColorCamera)
        self._xoutVideo = self._pipeline.create(dai.node.XLinkOut)

        self._xoutVideo.setStreamName("video")

        # Properties
        self._camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self._camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self._camRgb.setInterleaved(False)
        self._camRgb.setVideoSize(width, height)
        self._camRgb.setFps(fps)

        self._xoutVideo.input.setBlocking(False)
        self._xoutVideo.input.setQueueSize(1)

        # Linking
        self._camRgb.video.link(self._xoutVideo.input)

        # Connect to device and start pipeline
        self._device = dai.Device(self._pipeline)
        self._video = self._device.getOutputQueue(name="video", maxSize=1, blocking=False)
        self.fps_handler = FPSHandler()
        self.height = self._camRgb.getVideoHeight()
        self.width = self._camRgb.getVideoWidth()

    def get_color_frame(self, show_fps=False):
        video_in = self._video.get()
        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        if show_fps:
            self.fps_handler.next_iter()
            # return video_in.getCvFrame()
            return self.fps_handler.show_fps(video_in.getCvFrame(), round(self.fps_handler.fps(), 2))
        else:
            return video_in.getCvFrame()


if __name__ == '__main__':
    oak_d = OAK_D()
    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        cv2.imshow("VidraCar", frame)
        if cv2.waitKey(1) == ord('q'):
            break
