import cv2
import time
import threading


class Camera:
    """
    Threaded USB camera handler.
    Optimized for Raspberry Pi 3B+ and laptops.
    """

    def __init__(self, camera_id=0, resolution=(320, 240), fps=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()

    def initialize(self):
        """Initialize camera and start capture thread."""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Cannot open USB camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.5)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Continuously capture frames."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame.copy()
            else:
                with self.lock:
                    self.ret = False

    def read(self):
        """Return latest frame."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        """Stop thread and release camera."""
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
