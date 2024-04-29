import os
import sys
import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QProgressBar
from PyQt6.QtGui import QPixmap, QFont, QPainter, QPen, QColor, QCursor
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer

import shutil

IMAGE_DISPLAY_HEIGHT = 450

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Calculate the absolute path of the current working directory
        self.current_working_directory = os.getcwd()

        self.setWindowTitle("Image Selector")
        self.resize(1000, 800)  # Set initial window size

        main_layout = QVBoxLayout()

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)  # Fix progress bar height to 20
        self.progress_bar.setFixedWidth(800)   # Fix progress bar width to 800
        progress_layout = QHBoxLayout()  # Progress bar layout
        progress_layout.addWidget(self.progress_bar)

        main_layout.addLayout(progress_layout)  # Add progress bar layout

        # Large image left-aligned
        self.image_label = QLabel()
        self.image_label.setFixedHeight(IMAGE_DISPLAY_HEIGHT)  # Fix image height
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Align image to the left
        self.original_image_height = None  # placeholder, will be an int

        # Small image right-aligned
        self.small_image_label = QLabel()
        # self.small_image_label.setMaximumSize(300, IMAGE_DISPLAY_HEIGHT)

        # Create horizontal layout for large and small images
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.small_image_label)
        image_layout.addStretch(1)
        main_layout.addLayout(image_layout)  # Add horizontal layout to the main layout

        # Next button at the bottom
        self.next_button = QPushButton("NEXT")
        self.next_button.setFont(QFont("Arial", 12))  # Increase font size
        self.next_button.setStyleSheet("QPushButton { width: 150px; height: 50px; }")  # Increase button size
        self.next_button.clicked.connect(self.show_next_image)
        button_layout = QHBoxLayout()  # Use horizontal layout for buttons
        button_layout.addStretch(1)  # Add stretchable space before buttons
        button_layout.addWidget(self.next_button)
        button_layout.addStretch(1)  # Add stretchable space after buttons
        main_layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.image_files = []
        self.current_image_index = -1
        self.image_path = ""
        self.image_pixmap = None

        # variables to store the bounding boxes
        self.red_frame_params = []  # original image coords
        self.display_red_frame_params = []  # ui coords

        # Timer to track mouse position
        self.mouse_position = QPoint()
        self.mouse_timer = QTimer()
        self.mouse_timer.timeout.connect(self.track_mouse_position)
        self.mouse_timer.start(50)  # Set interval to 50 milliseconds, approximately 20 frames per second
        self.hovered_red_frame_index = None

        # Read data from json file
        with open("output.json", "r") as f:
            self.data = json.load(f)
            print(self.data)

        # Load information of the first element
        self.load_image_info(0)

    def load_image_info(self, index):
        if index < len(self.data):
            image_info = self.data[index]
            self.image_path = image_info["image_path"]
            self.image_pixmap = QPixmap(image_info["image_path"])

            # load bounding boxes both at original scale and at display scale
            # display scale will be used for drawing, and comparison with the mouse position
            self.red_frame_params = image_info["bboxes"][:10]
            original_image_height = self.image_pixmap.height()
            rescale_func = lambda x: int(x * IMAGE_DISPLAY_HEIGHT / original_image_height)
            self.display_red_frame_params = list(map(lambda box: list(map(rescale_func, box)), self.red_frame_params))

            self.current_image_index = index
            self.update_progress_bar()
            self.show_image()

    def update_progress_bar(self):
        if self.data:
            progress_value = int((self.current_image_index + 1) / len(self.data) * 100)
            self.progress_bar.setValue(progress_value)

    def show_image(self):
        if self.current_image_index >= 0 and self.current_image_index < len(self.data):
            # rescale image to a reasonable display height
            scaled_pixmap = self.image_pixmap.scaledToHeight(IMAGE_DISPLAY_HEIGHT, Qt.TransformationMode.SmoothTransformation)

            # Draw red frames using a QPainter
            painter = QPainter(scaled_pixmap)
            for i, frame_params in enumerate(self.display_red_frame_params):
                stroke_color = QColor("orange") if i == self.hovered_red_frame_index else QColor("red")
                stroke_width = 4 if i == self.hovered_red_frame_index else 2
                pen = QPen(stroke_color, stroke_width)
                painter.setPen(pen)
                painter.drawRect(*frame_params)
            painter.end()

            # Display image with red frames drawn
            self.image_label.setPixmap(scaled_pixmap)

    def show_zoom_in_image(self, red_frame_index):
        # If no index, clear the small image
        if red_frame_index is None:
            self.small_image_label.clear()
            return
        # Crop image to the red frame
        red_frame = self.red_frame_params[red_frame_index]
        small_pixmap = self.image_pixmap.copy(*red_frame)
        # Scale to a reasonable size
        small_pixmap = small_pixmap.scaled(300, IMAGE_DISPLAY_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.small_image_label.setPixmap(small_pixmap)

    def show_next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.data):
            self.current_image_index = 0
        self.load_image_info(self.current_image_index)

    def track_mouse_position(self):        
        # Get mouse click position (relative to the window)
        self.mouse_position = self.mapFromGlobal(QCursor.pos())

        # Store the indexes of red frames where mouse position is located
        inside_red_frames = []

        # Calculate the positions and ranges of red frames on the screen, and check if the mouse position is inside
        for index, red_frame_params in enumerate(self.display_red_frame_params):
            # Position and size of red frame in the window
            red_frame_rect = QRect(*red_frame_params)

            # Position and range of red frame on the screen
            red_frame_screen_rect = red_frame_rect.translated(self.image_label.pos())

            # If mouse position is inside the red frame, add its index to the list
            if red_frame_screen_rect.contains(self.mouse_position):
                inside_red_frames.append(index)

        # If mouse position is inside multiple red frames, select the one with the smallest area
        if len(inside_red_frames) > 1:
            smallest_area = float('inf')
            selected_red_frame = None
            for index in inside_red_frames:
                red_frame_params = self.red_frame_params[index]
                red_frame_area = red_frame_params[2] * red_frame_params[3]  # Calculate the area of red frame
                if red_frame_area < smallest_area:
                    smallest_area = red_frame_area
                    selected_red_frame = index
            inside_red_frames = [selected_red_frame]

        # Store the red frame where the mouse position is located
        if inside_red_frames:
            # print(f"Mouse position is inside red frame {inside_red_frames[0] + 1}")
            self.hovered_red_frame_index = inside_red_frames[0]
        else:
            # print("Mouse position is not inside any red frame")
            self.hovered_red_frame_index = None
        
        # Update the main view (to show hovered box)
        self.show_image()

        # Update the zoom in view
        self.show_zoom_in_image(self.hovered_red_frame_index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
