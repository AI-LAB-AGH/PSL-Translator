import cv2
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

class VideoCapture(QObject):
    frame_captured = pyqtSignal(object)

    def __init__(self, update_interval=30):
        super().__init__()
        self.capture = cv2.VideoCapture(0) 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(update_interval)
        self.show_text = False
        self.recognized_text = ""

    def update_frame(self):
        ret, frame = self.capture.read()
        
        if not ret:
            return
        
        self.frame_captured.emit(frame)
        
    def set_show_text(self, value: bool):
        self.show_text = value
    
    def set_recognized_text(self, text: str):
        self.recognized_text = text

    def release(self):
        self.capture.release()
