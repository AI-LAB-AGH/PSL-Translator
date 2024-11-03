from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea
from PyQt5.QtCore import Qt, QRect, QPropertyAnimation
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QFont, QFontDatabase

class SidePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        id = QFontDatabase.addApplicationFont("app/assets/InriaSans-Regular.ttf")
        families = QFontDatabase.applicationFontFamilies(id)
        if families:
            custom_font_family = families[0]
            custom_font = QFont(custom_font_family)
        else:
            print("Failed to load the font.")
            custom_font = QFont()
            
        self.setFixedWidth(600)
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title = QLabel("Content Placeholder")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(custom_font)
        
        self.content = QLabel("Content Placeholder")
        self.content.setWordWrap(True)
        self.content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.content.setAlignment(Qt.AlignTop)
        self.content.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content.setFont(custom_font)
        
        self.scroll_layout.addWidget(self.title, stretch=0)
        self.scroll_layout.addWidget(self.content, stretch=1)
        
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)

        self.hide()
        
        self.main_layout.setObjectName("side_panel")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_content.setObjectName("scroll_content")
        self.title.setObjectName("title")
        self.content.setObjectName("content")
        
        # TODO: move to stylesheet
        self.setStyleSheet("""
            #side_panel {
                background-color: #132234;
                color: white;
                margin: 0;
                padding-left: 40px;
            }
            
            #scroll_area {
                background-color: transparent;
                border: none;
                margin: 0;
            }
            
            #scroll_content {
                background-color: #132234;
                color: white;
                margin: 0;
            }
            
            #title {
                font-size: 24px;
                font-weight: bold;
                color: white;
                padding: 25px;
                padding-top: 40px;
            }
        
            #content {
                font-size: 16px;
                color: white;
                line-height: 1.5;
                padding: 25px;
            }
        """)

    def set_content(self, content):
        self.title.setText(content)
        match content:
            case "User instruction":
                with open('app/side_panels_content/user_instruction.txt', 'r') as file:
                    self.content.setText(self.wrap_text(file.read()))
            case "Currently available gestures":
                with open('app/side_panels_content/gestures_list.txt', 'r') as file:
                    self.content.setText(self.wrap_text(file.read()))
            case "About the project":
                with open('app/side_panels_content/project_description.txt', 'r') as file:
                    self.content.setText(self.wrap_text(file.read()))

    def wrap_text(self, text):
        return f'<div style="text-align: justify;">{text}</div>'

    def resizeEvent(self, event):
        self.setFixedHeight(self.parent().height())
        super().resizeEvent(event)

    def show_panel(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(QRect(-600, 0, 600, self.parent().height()))
        self.animation.setEndValue(QRect(375, 0, 600, self.parent().height()))
        self.animation.start()
        self.show()

    def hide_panel(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(QRect(375, 0, 600, self.parent().height()))
        self.animation.setEndValue(QRect(-600, 0, 600, self.parent().height()))
        self.animation.finished.connect(self.hide)
        self.animation.start()
