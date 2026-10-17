from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog
)
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtMultimedia import QMediaPlayer, QVideoFrame, QVideoSink
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QSize, Qt, QUrl
import numpy as np
from ultralytics import YOLO
import cv2
import time

class PeopleTracker:
    def __init__(self):
        super().__init__()
        self.model = YOLO("yolo26s.pt") 

    def track(self, image):
        result = self.model.track(image, persist=True, classes=[0], verbose=False)[0]
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            return  boxes, ids
        
        return [], []


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_width = 1080
        self.video_height = 720
        self.last_frame_time = time.time()

        self.setWindowTitle("Video Player")
        self.resize(self.video_width + 20, self.video_height + 20)

        self.player = QMediaPlayer()
        self.video_widget = QLabel()
        self.video_widget.setScaledContents(True)
        self.video_sink = QVideoSink()
        self.player.setVideoSink(self.video_sink)

        self.people_tracker = PeopleTracker()

        self.video_sink.videoFrameChanged.connect(
            self.on_frame_changed, 
            Qt.ConnectionType.QueuedConnection
        )

        self.open_btn = QPushButton("Open file")
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")

        self.open_btn.clicked.connect(self.open_file)
        self.play_btn.clicked.connect(self.player.play)
        self.pause_btn.clicked.connect(self.player.pause)
        self.stop_btn.clicked.connect(self.player.stop)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)

        self.time_label = QLabel("00:00 / 00:00")

        video = QHBoxLayout()
        video.addStretch()
        video.addWidget(self.video_widget)
        video.addStretch()

        controls = QHBoxLayout()
        controls.addWidget(self.open_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.position_slider, stretch=1)
        controls.addWidget(self.time_label)


        main_layout = QVBoxLayout()
        main_layout.addLayout(video, stretch=1)
        main_layout.addLayout(controls)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.handle_error)

    def on_frame_changed(self, frame: QVideoFrame):

        if not frame.isValid():
            return

        if  time.time() - self.last_frame_time < 0.2:
            return

        self.last_frame_time = time.time()

        start_time = time.time()
        image = frame.toImage()

        if image.format() not in (QImage.Format.Format_RGB32, QImage.Format.Format_ARGB32):
            image = image.convertToFormat(QImage.Format.Format_RGB32)

        ptr = image.constBits()
        ptr.setsize(image.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.width(), 4)
        bgr_array = arr[..., :3].copy()
        boxes, ids = self.people_tracker.track(bgr_array)

        painter = QPainter(image) 
        for box, id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            painter.drawRect(x1, y1, x2, y2)
        painter.end()

        pixmap = QPixmap.fromImage(image)
        sz = pixmap.size()
        scaleX = scaleY = 1.0

        if sz.width() > self.video_width:
            scaleX = self.video_width / sz.width()

        if sz.height() > self.video_height:
            scaleY = self.video_height / sz.height()

        sz = QSize(round(min(scaleX, scaleY) * sz.width()),
                   round(min(scaleX, scaleY) * sz.height()))

        self.video_widget.setPixmap(pixmap)
        self.video_widget.setFixedSize(sz)
        print(time.time() - start_time, len(boxes))
        
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)"
        )
        if filepath:
            self.player.setSource(QUrl.fromLocalFile(filepath))
            self.player.play()

    def position_changed(self, position):
        self.position_slider.setValue(position)
        self.update_time_label()

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.update_time_label()

    def update_time_label(self):
        pos = self.player.position() // 1000
        dur = self.player.duration() // 1000
        pos_m, pos_s = divmod(pos, 60)
        dur_m, dur_s = divmod(dur, 60)
        self.time_label.setText(
            f"{pos_m:02d}:{pos_s:02d} / {dur_m:02d}:{dur_s:02d}"
        )

    def handle_error(self, error):
        print(f"Ошибка медиаплеера: {error}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec())
