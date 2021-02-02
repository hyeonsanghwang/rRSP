from camera.video_stream import VideoStream
import cv2

CAMERA_NUM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

stream = VideoStream(CAMERA_NUM, FPS, FRAME_WIDTH, FRAME_HEIGHT)
while True:
    ret, frame = stream.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(stream.delay())
    if key == 27:
        break

cv2.destroyAllWindows()
stream.release()