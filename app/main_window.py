from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QDesktopWidget, QSizePolicy, QPushButton, QSpacerItem, QGraphicsDropShadowEffect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from app.components.video_capture import VideoCapture
from app.components.menu_component import MenuComponent
import cv2
from app.components.side_panel import SidePanel
from app.helpers.format_sentence import format_sentence
from PyQt5.QtGui import QFont, QFontDatabase
from app.assets.shadow_effect import shadow_effect

class MainWindow(QMainWindow):
    def __init__(self, prediction_handler, transform):
        super().__init__()
        id = QFontDatabase.addApplicationFont("app/assets/InriaSans-Regular.ttf")
        families = QFontDatabase.applicationFontFamilies(id)
        if families:
            custom_font_family = families[0]
            self.custom_font = QFont(custom_font_family)
        else:
            self.custom_font = QFont()
            
        self.prediction_handler = prediction_handler
        self.transform = transform
        self.setWindowTitle("Polish Sign Language Translator")
        self.setGeometry(100, 100, 1200, 800)
        self.center_window()

        self.init_ui()

        self.video_capture = VideoCapture()
        self.video_capture.frame_captured.connect(self.update_video_label)

        self.is_running = False

        self.reset_text_timer = QTimer()
        self.reset_text_timer.setInterval(2000)
        self.reset_text_timer.timeout.connect(self.reset_recognized_text)

        self.recognized_gestures = []
        self.current_output = []
        
    def init_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setGraphicsEffect(shadow_effect())

        self.start_stop_button = QPushButton("START", self)
        self.start_stop_button.clicked.connect(self.toggle_start_stop)
        self.start_stop_button.setFont(self.custom_font)
        self.start_stop_button.setGraphicsEffect(shadow_effect())

        self.start_stop_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_stop_button.setMaximumWidth(150)

        self.sentence_label = QLabel(self)
        self.sentence_label.setAlignment(Qt.AlignCenter)
        self.custom_font.setPointSize(12)
        self.sentence_label.setFont(self.custom_font)
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.menu_list = MenuComponent(self)
        self.menu_list.option_clicked.connect(self.show_side_panel)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.start_stop_button)
        button_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.addStretch(1)
        right_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        right_layout.addLayout(button_layout)
        right_layout.addWidget(self.sentence_label)
        right_layout.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.menu_list, alignment=Qt.AlignLeft)
        main_layout.addStretch(1)
        main_layout.addLayout(right_layout)
        main_layout.addStretch(1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.side_panel = SidePanel(self)

    def center_window(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        window_geometry = self.frameGeometry()
        screen_center = screen_geometry.center()
        window_geometry.moveCenter(screen_center)
        self.move(window_geometry.topLeft())

    def update_video_label(self, frame):
        mirrored_frame = cv2.flip(frame, 1)
        if self.is_running:
            recognized_action, confidence, output = self.prediction_handler.process_frame(frame)
            self.current_output = output
            # print('output: ', output)
            # print('self.current_output: ', self.current_output)
            
            if recognized_action is not None:
                self.recognized_gestures.append(f"{recognized_action}")
                self.video_capture.set_recognized_text(recognized_action)
                
                self.reset_text_timer.start()

            if self.video_capture.recognized_text:
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                position = (mirrored_frame.shape[1] - 10 - cv2.getTextSize(self.video_capture.recognized_text, font, font_scale, thickness)[0][0], 30)
                font_scale = 1
                color = (255, 255, 255)
                cv2.putText(mirrored_frame, self.video_capture.recognized_text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        mirrored_frame_rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = mirrored_frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(mirrored_frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def reset_recognized_text(self):
        self.video_capture.set_recognized_text("")

    def toggle_start_stop(self):
        self.is_running = not self.is_running
        self.start_stop_button.setText("STOP" if self.is_running else "START")
        self.video_capture.set_show_text(True if self.is_running else False)
        
        if self.is_running:
            self.recognized_gestures.clear()
            self.current_output.clear()
            self.sentence_label.setText("")
            self.reset_text_timer.start()
        else:
            # TODO: Translate sentence
            print('formated: ', format_sentence(self.current_output))
            self.sentence_label.setText(format_sentence(self.current_output))

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)
        
    def show_side_panel(self, content):
        if content == "":
            self.side_panel.hide_panel()
            return
        if self.side_panel.isVisible():
            if self.side_panel.title.text() == content:
                self.side_panel.hide_panel()
            else:
                self.is_switching_panel_content = True
                self.side_panel.hide_panel()
                QTimer.singleShot(300, lambda: self.set_new_panel_content(content))
        else:
            self.side_panel.set_content(content)
            self.side_panel.show_panel()

    def set_new_panel_content(self, content):
        if self.is_switching_panel_content:
            self.side_panel.set_content(content)
            self.side_panel.show_panel()
            self.is_switching_panel_content = False
