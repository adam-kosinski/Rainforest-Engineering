import os
import sys
import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QProgressBar, QFileDialog, QFrame
from PyQt6.QtGui import QPixmap, QFont, QPainter, QPen, QColor, QCursor
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer
from generate_crops import create_zoom_out_crops
import threading

IMAGE_DISPLAY_HEIGHT = 450


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Selector")
        self.resize(1100, IMAGE_DISPLAY_HEIGHT + 240)  # Set initial window size

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # json file selector
        file_chooser_layout = QHBoxLayout()
        file_chooser_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        json_file_chooser = QPushButton("Select JSON File From Pipeline Output")
        json_file_chooser.clicked.connect(self.open_json_file_dialog)
        file_chooser_layout.addWidget(json_file_chooser)
        file_chooser_layout.addSpacing(5)
        self.json_file_label = QLabel()
        file_chooser_layout.addWidget(self.json_file_label)
        main_layout.addLayout(file_chooser_layout)

        # widget that starts as hidden, but once a json file is selected it is shown
        # most of the UI stuff is its layout
        self.when_loaded_widget = QWidget()
        self.when_loaded_widget.hide()
        when_loaded_layout = QVBoxLayout()
        self.when_loaded_widget.setLayout(when_loaded_layout)
        main_layout.addWidget(self.when_loaded_widget)

        # save directory location
        save_directory_layout = QHBoxLayout()
        save_directory_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # display
        self.save_directory_label = QLabel()
        save_directory_layout.addWidget(self.save_directory_label)
        # button to change the save path
        change_save_directory_button = QPushButton("Change")
        change_save_directory_button.clicked.connect(self.open_save_directory_dialog)
        save_directory_layout.addWidget(change_save_directory_button)
        when_loaded_layout.addLayout(save_directory_layout)
        horiz_line = QFrame()
        horiz_line.setFrameShape(QFrame.Shape.HLine)
        horiz_line.setFrameShadow(QFrame.Shadow.Sunken)
        when_loaded_layout.addWidget(horiz_line)

        # header indicating the current image with progress bar
        progress_layout = QHBoxLayout()
        # image name
        self.image_name_label = QLabel()
        self.image_name_label.setFont(QFont("Arial", 12))  # Increase font size
        progress_layout.addWidget(self.image_name_label)
        progress_layout.addSpacing(10)
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(400)   # Fix progress bar width to 800
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addStretch(1)
        # add to main layout
        when_loaded_layout.addLayout(progress_layout)

        # Instructions
        instructions = QLabel("Hover over a box to see zoomed in view on the right. Click a box to toggle whether it is selected. Click 'SAVE AND NEXT' to generate zoom out sequences for all selected boxes, which are saved in the output directory shown above.")
        instructions.setMaximumWidth(650)
        instructions.setWordWrap(True)
        when_loaded_layout.addWidget(instructions)

        # Large image left-aligned
        self.image_label = QLabel()
        self.image_label.setFixedHeight(IMAGE_DISPLAY_HEIGHT)  # Fix image height
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Align image to the left
        self.image_label.mousePressEvent = self.handle_image_click

        # Small image right-aligned
        self.small_image_label = QLabel()

        # Create horizontal layout for large and small images
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addSpacing(10)
        image_layout.addWidget(self.small_image_label)
        when_loaded_layout.addLayout(image_layout)  # Add horizontal layout to the main layout

        # Back button
        back_button = QPushButton("BACK")
        back_button.setFixedSize(100, 50)
        back_button.setFont(QFont("Arial", 12))  # Increase font size
        back_button.clicked.connect(self.show_previous_image)

        # Skip button at the bottom
        skip_button = QPushButton("SKIP")
        skip_button.setFixedSize(100, 50)
        skip_button.setFont(QFont("Arial", 12))  # Increase font size
        skip_button.clicked.connect(lambda: self.show_next_image(save_crops=False))

        # Save and next button at the bottom
        next_button = QPushButton("SAVE AND NEXT")
        next_button.setFixedSize(200, 50)
        next_button.setFont(QFont("Arial", 12))  # Increase font size
        next_button.clicked.connect(lambda: self.show_next_image(save_crops=True))

        button_layout = QHBoxLayout()  # Use horizontal layout for buttons
        button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(back_button)
        button_layout.addSpacing(50)
        button_layout.addWidget(skip_button)
        button_layout.addWidget(next_button)
        # button_layout.addStretch(1)  # Add stretchable space after buttons
        when_loaded_layout.addLayout(button_layout)

        # image variables
        self.image_files = []
        self.current_image_index = -1
        self.image_path = ""
        self.image_pixmap = None

        # where to save crops
        self.save_directory = ""

        # variables to store the bounding boxes
        self.red_frame_params = []  # original image coords
        self.display_red_frame_params = []  # ui coords

        # Timer to track mouse position
        self.mouse_position = QPoint()
        self.mouse_timer = QTimer()
        self.mouse_timer.timeout.connect(self.track_mouse_position)
        self.mouse_timer.start(50)  # Set interval to 50 milliseconds, approximately 20 frames per second

        # Selection stuff
        self.hovered_red_frame_index = None
        self.selected_frame_indices = []

        # Variable storing json data that came from running the pipeline
        self.data = []
        self.load_json_file("output.json")
    
    def open_json_file_dialog(self):
        json_file, _ = QFileDialog.getOpenFileName(self, "Select JSON File From Pipeline Output", "", "JSON Files (*.json)")
        if json_file:
            self.load_json_file(json_file)

    def load_json_file(self, json_file):
        with open(json_file) as f:
            self.data = json.load(f)
        if len(self.data) > 0:
            self.json_file_label.setText("Current JSON file -- " + os.path.abspath(json_file))
            self.save_directory = os.path.join(os.getcwd(), "zoom_sequences")
            self.save_directory_label.setText(f"Output directory -- {self.save_directory}")
            self.load_image_info(0)
            self.when_loaded_widget.show()
    
    def open_save_directory_dialog(self):
        new_save_dir = QFileDialog.getExistingDirectory(self, "Select Folder To Save Crops In", "")
        if new_save_dir:
            self.save_directory = new_save_dir
            self.save_directory_label.setText(f"Output directory -- {self.save_directory}")

    def load_image_info(self, index):
        if index < len(self.data):
            image_info = self.data[index]
            self.image_path = image_info["image_path"]
            self.image_pixmap = QPixmap(image_info["image_path"])

            # load bounding boxes (aka red frames) both at original scale and at display scale
            # display scale will be used for drawing, and comparison with the mouse position
            self.red_frame_params = image_info["bboxes"][:5]
            original_image_height = self.image_pixmap.height()
            rescale_func = lambda x: int(x * IMAGE_DISPLAY_HEIGHT / original_image_height)
            self.display_red_frame_params = list(map(lambda box: list(map(rescale_func, box)), self.red_frame_params))

            # by default, all frames are selected
            self.selected_frame_indices = list(range(len(self.red_frame_params)))

            self.current_image_index = index
            self.image_name_label.setText(f"Current image: {os.path.basename(self.image_path)}")
            self.update_progress_bar()
            self.show_image()

    def update_progress_bar(self):
        if self.data:
            progress_value = int((self.current_image_index + 1) / len(self.data) * 100)
            self.progress_bar.setValue(progress_value)
            self.progress_bar.setFormat(f"Image {self.current_image_index + 1}/{len(self.data)}")

    def show_image(self):
        if self.current_image_index >= 0 and self.current_image_index < len(self.data):
            # rescale image to a reasonable display height
            image_pixmap = self.image_pixmap.scaledToHeight(IMAGE_DISPLAY_HEIGHT, Qt.TransformationMode.SmoothTransformation)

            # Create an overlay pixmap that darkens the non selected areas
            overlay_pixmap = QPixmap(image_pixmap.size())
            overlay_pixmap.fill(QColor(0, 0, 0, 170))
            overlay_painter = QPainter(overlay_pixmap)
            # change composition mode and opacity so that new rects will be drawn transparent and replace existing pixels
            overlay_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            overlay_painter.setOpacity(0.0)
            for i, (x,y,w,h) in enumerate(self.display_red_frame_params):
                if i in self.selected_frame_indices:
                    overlay_painter.fillRect(x, y, w, h, Qt.GlobalColor.black)
            overlay_painter.end()

            # Use a QPainter to draw on the pixmap, draw the image and the darkener overlay
            display_pixmap = QPixmap(image_pixmap.size())
            painter = QPainter(display_pixmap)
            painter.drawPixmap(display_pixmap.rect(), image_pixmap, image_pixmap.rect())
            painter.drawPixmap(display_pixmap.rect(), overlay_pixmap, overlay_pixmap.rect())
            # Draw red frames
            for i, frame_params in enumerate(self.display_red_frame_params):
                # determine which pen to use - different for hovering, selected, and deselected
                if i == self.hovered_red_frame_index:
                    pen = QPen(QColor("orange"), 4)
                elif i in self.selected_frame_indices:
                    pen = QPen(QColor("red"), 2)
                else:
                    pen = QPen(QColor("red"), 2, Qt.PenStyle.DotLine)
                painter.setPen(pen)
                painter.drawRect(*frame_params)
            painter.end()

            # Display image with red frames drawn
            self.image_label.setPixmap(display_pixmap)

    def show_zoom_in_image(self, red_frame_index):
        # If no index, clear the small image
        if red_frame_index is None:
            self.small_image_label.clear()
            return
        # Crop image to the red frame
        red_frame = self.red_frame_params[red_frame_index]
        small_pixmap = self.image_pixmap.copy(*red_frame)
        # Scale to a reasonable size
        small_pixmap = small_pixmap.scaled(400, IMAGE_DISPLAY_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.small_image_label.setPixmap(small_pixmap)

    def show_next_image(self, save_crops=True):
        if len(self.data) == 0:
            return
        if save_crops:
            self.save_zoom_sequences()
        self.load_image_info((self.current_image_index + 1) % len(self.data))

    def show_previous_image(self):
        if len(self.data) == 0:
            return
        self.load_image_info((self.current_image_index - 1) % len(self.data))

    def track_mouse_position(self):
        # Get mouse position (relative to the big image)
        self.mouse_position = self.image_label.mapFromGlobal(QCursor.pos())

        # Store the indexes of red frames where mouse position is located
        inside_red_frames = []

        # Calculate the positions and ranges of red frames on the screen, and check if the mouse position is inside
        for index, red_frame_params in enumerate(self.display_red_frame_params):
            # Position and size of red frame in the big image
            red_frame_rect = QRect(*red_frame_params)

            # If mouse position is inside the red frame, add its index to the list
            if red_frame_rect.contains(self.mouse_position):
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
    
    def handle_image_click(self, event):
        if self.hovered_red_frame_index in self.selected_frame_indices:
            self.selected_frame_indices.remove(self.hovered_red_frame_index)
        else:
            self.selected_frame_indices.append(self.hovered_red_frame_index)
    
    def save_zoom_sequences(self):
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        save_directory = os.path.join(self.save_directory, image_name)
        boxes_to_save = []
        for i, box in enumerate(self.red_frame_params):
            if i in self.selected_frame_indices:
                boxes_to_save.append(box)

        # save images on another thread, so the UI doesn't freeze briefly
        thread = threading.Thread(target=create_zoom_out_crops, args=(self.image_path, boxes_to_save, save_directory))
        thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
