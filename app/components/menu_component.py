from PyQt5.QtWidgets import QListWidget, QLabel, QWidget, QVBoxLayout, QListWidgetItem, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QPainter, QPixmap
from app.assets.shadow_effect import shadow_effect, bottom_shadow_effect

class MenuComponent(QWidget):
    option_clicked = pyqtSignal(str)
    
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

        self.title_widget = QLabel("Polish Sign Language")
        self.title_widget.setAlignment(Qt.AlignLeft)
        self.title_widget.setFixedWidth(375)
        self.title_widget.setFont(custom_font)
        self.title_widget.setGraphicsEffect(shadow_effect())
        
        self.subtitle_widget = QLabel("Translator")
        self.subtitle_widget.setAlignment(Qt.AlignLeft)
        self.subtitle_widget.setFixedWidth(375)
        self.subtitle_widget.setFont(custom_font)
        self.subtitle_widget.setGraphicsEffect(bottom_shadow_effect())
        
        self.main_menu = QListWidget()
        self.main_menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_menu.setFont(custom_font)
        
        self.add_menu_option("User instruction", self.on_user_instruction_click)
        self.add_menu_option("Currently available gestures", self.on_gestures_click)
        self.add_menu_option("About the project", self.on_about_project_click)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.title_widget)
        self.layout.addWidget(self.subtitle_widget)
        self.layout.addWidget(self.main_menu)
        self.setLayout(self.layout)
        
        self.title_widget.setObjectName("title_widget")
        self.subtitle_widget.setObjectName("subtitle_widget")
        self.main_menu.setObjectName("main_menu")
        self.setObjectName("MenuComponent")
        
        self.last_selected_item = None
        
        self.main_menu.itemClicked.connect(self.handle_item_click)

    def add_menu_option(self, text, callback):
        item = QListWidgetItem(text)
        self.main_menu.addItem(item)
        item.setData(Qt.UserRole, callback)
      
    def handle_item_click(self, selected_item):
        if self.last_selected_item == selected_item:
            self.main_menu.clearSelection()
            self.last_selected_item = None
            self.option_clicked.emit("")
        else:
            self.last_selected_item = selected_item
            callback = selected_item.data(Qt.UserRole)
            if callable(callback):
                callback()
            self.option_clicked.emit(selected_item.text())

    def on_user_instruction_click(self):
        pass

    def on_gestures_click(self):
        pass

    def on_about_project_click(self):
        pass
    
