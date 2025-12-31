"""
RealSense Camera Application with YOLOv11 Detection and 3D Measurement
RealSense D435 streaming, object detection, and depth-based measurement
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging

from functools import lru_cache
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QGroupBox,
    QTextEdit,
    QSpinBox,
    QFormLayout,
    QMessageBox,
    QStatusBar,
    QRubberBand,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Avoid importing Ultralytics/torch at module import time.
YOLO = None  # resolved at runtime when loading model

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return p

@lru_cache(maxsize=1)
def load_yolo_model_cached(model_path: str, device: str):
    """Load and cache the YOLO model to avoid repeated imports."""
    try:
        from ultralytics import YOLO as _YOLO
    except ImportError:
        logger.warning("Ultralytics not available - object detection will be skipped")
        return None

    resolved = _resolve_model_path(model_path)
    if not resolved.exists():
        logger.error(f"YOLO model not found at: {resolved}")
        return None

    logger.info(f"Loading YOLO model from {resolved}")
    model = _YOLO(str(resolved))
    try:
        model.to(device)
        logger.info(f"YOLO model loaded on {device}")
    except Exception as e:
        logger.warning(f"Could not move YOLO model to {device}: {e}")
    return model


@dataclass
class CameraIntrinsics:
    """Store camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1280
    height: int = 720


@dataclass
class MeasurementResult:
    """Store 3D measurement results"""
    diameter_mm: Optional[float] = None
    height_mm: Optional[float] = None
    center_3d: Optional[Tuple[float, float, float]] = None
    confidence: float = 0.0
    error_message: Optional[str] = None


class RealSenseStreamer:
    """Handle RealSense camera streaming in a separate thread"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.depth_scale = 0.001
        
        self.is_running = False
        self.stop_flag = False
        
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.color_frame = None
        self.depth_frame = None
    
    def initialize(self) -> bool:
        """Initialize RealSense pipeline"""
        try:
            logger.info("Initializing RealSense camera...")
            
            # Check for connected devices
            ctx = rs.context()
            if ctx.query_devices().size() == 0:
                msg = "No RealSense devices detected"
                logger.error(msg)
                return False
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start pipeline
            self.profile = self.pipeline.start(self.config)
            
            # Create alignment object (align depth to color)
            self.align = rs.align(rs.stream.color)
            
            # Configure depth sensor
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Set high accuracy preset
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
            
            # Set maximum laser power
            if depth_sensor.supports(rs.option.laser_power):
                max_power = depth_sensor.get_option_range(rs.option.laser_power).max
                depth_sensor.set_option(rs.option.laser_power, max_power)
                logger.info(f"Laser power set to: {max_power}")
            
            # Get camera intrinsics
            # NOTE: We get intrinsics from the COLOR stream, but since we aligned depth to color,
            # these intrinsics are in the DEPTH coordinate frame. This is what rs2_deproject_pixel_to_point needs.
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            
            self.intrinsics = CameraIntrinsics(
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.ppx,
                cy=intr.ppy,
                width=intr.width,
                height=intr.height
            )
            
            logger.info(f"Camera intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            logger.info(f"Resolution: {self.intrinsics.width}x{self.intrinsics.height}")
            logger.info(f"Depth scale: {self.depth_scale}")
            
            return True
            
        except Exception as e:
            msg = f"Error initializing RealSense: {e}"
            logger.error(msg)
            return False
    
    def run(self):
        """Streaming thread loop"""
        if not self.initialize():
            return
        
        self.is_running = True
        self.stop_flag = False
        
        try:
            while not self.stop_flag:
                try:
                    # Wait for frames (tolerate occasional timeouts)
                    frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                    
                    # Align frames
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    # Convert to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    self.color_frame = color_frame
                    self.depth_frame = depth_frame
                    
                except RuntimeError as e:
                    msg = str(e)
                    # Suppress noisy timeout logs; just continue
                    if "Frame didn't arrive" in msg or "processing block" in msg:
                        continue
                    logger.error(f"Error in streaming loop: {e}")
                except Exception as e:
                    logger.error(f"Error in streaming loop: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.pipeline:
                self.pipeline.stop()
            self.is_running = False
            logger.info("RealSense pipeline stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class YOLODetector:
    """Handle YOLOv11 object detection"""
    
    def __init__(self, model_path: str = "yolov11n.pt"):
        """Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
        """
        self.model = None
        self.model_path = model_path
        self.device = "cuda" if _has_cuda() else "cpu"

        # Lazy load + cache model
        self.model = load_yolo_model_cached(self.model_path, self.device)
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """Run YOLO detection on frame
        
        Args:
            frame: Input image frame (BGR)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with detection results including masks
        """
        if self.model is None:
            logger.warning("YOLO model not available")
            return {"detections": [], "mask": None}
        
        try:
            # Run inference
            results = self.model(frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            detections = []
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            # Extract detections with masks
            if result.masks is not None:
                for i, (box, mask_tensor) in enumerate(zip(result.boxes.data, result.masks.data)):
                    mask_numpy = mask_tensor.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask_numpy, (frame.shape[1], frame.shape[0]))
                    
                    detections.append({
                        "box": box.cpu().numpy(),
                        "mask": mask_resized,
                        "confidence": float(box[4].cpu().numpy())
                    })
                    
                    # Combine masks
                    mask = cv2.bitwise_or(mask, mask_resized)
            
            return {
                "detections": detections,
                "mask": mask if np.any(mask) else None,
                "raw_results": result
            }
        
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return {"detections": [], "mask": None}


class DepthProcessor:
    """Handle depth-based 3D measurements"""
    
    def __init__(self, intrinsics: CameraIntrinsics, depth_scale: float = 0.001):
        """Initialize depth processor
        
        Args:
            intrinsics: Camera intrinsic parameters
            depth_scale: Depth scale factor (mm per unit)
        """
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
    
    def deproject_pixel_to_3d(self, x: float, y: float, depth: float) -> Tuple[float, float, float]:
        """Convert 2D pixel to 3D world coordinates using depth
        
        Uses RealSense depth intrinsics (in depth coordinate frame) to deproject a 2D pixel
        coordinate with its corresponding depth value into 3D world coordinates.
        
        Args:
            x: Pixel x coordinate (in depth/color aligned frame)
            y: Pixel y coordinate (in depth/color aligned frame)
            depth: Depth value in raw units (converted to meters for rs2_deproject_pixel_to_point)
            
        Returns:
            3D point (x, y, z) in millimeters
            
        Note:
            The intrinsics stored in self.intrinsics are DEPTH frame intrinsics (from aligned color stream).
            This is correct for rs2_deproject_pixel_to_point which requires depth intrinsics.
        """
        try:
            # Create intrinsics object for pyrealsense2
            intr = rs.intrinsics()
            intr.fx = self.intrinsics.fx
            intr.fy = self.intrinsics.fy
            intr.ppx = self.intrinsics.cx
            intr.ppy = self.intrinsics.cy
            intr.width = self.intrinsics.width
            intr.height = self.intrinsics.height
            intr.model = rs.distortion.none
            intr.coeffs = [0, 0, 0, 0, 0]
            
            # Deproject pixel to 3D (RealSense expects depth in meters)
            depth_m = depth * self.depth_scale / 1000.0
            point_3d = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth_m)
            
            # Convert to mm
            return (point_3d[0] * 1000, point_3d[1] * 1000, point_3d[2] * 1000)
        
        except Exception as e:
            logger.error(f"Error deprojecting pixel: {e}")
            return (0, 0, 0)
    
    def get_depth_at_point(self, x: int, y: int, depth_frame: np.ndarray, 
                          window_size: int = 15) -> float:
        """Get valid depth value in a window around specified point
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            depth_frame: Depth image array
            window_size: Size of window for averaging
            
        Returns:
            Depth value in raw units
        """
        h, w = depth_frame.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        half_window = window_size // 2
        x_start = max(x - half_window, 0)
        x_end = min(x + half_window + 1, w)
        y_start = max(y - half_window, 0)
        y_end = min(y + half_window + 1, h)
        
        window = depth_frame[y_start:y_end, x_start:x_end]
        valid_depths = window[(window > 0) & (window < 65535)]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        return 0.0
    
    def measure_height_from_mask(self, mask: np.ndarray, depth_frame: np.ndarray) -> MeasurementResult:
        """Measure object height from mask and depth map
        
        Args:
            mask: Binary mask of detected object
            depth_frame: Depth image array
            
        Returns:
            MeasurementResult with height measurement
        """
        result = MeasurementResult()
        
        try:
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                result.error_message = "No contours found in mask"
                return result
            
            # Get largest contour
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Use center 40% of mask for height calculation
            center_start_x = x + int(w * 0.30)
            center_end_x = x + int(w * 0.70)
            
            # Get all mask pixels in center region
            ys, xs = np.where(mask[y:y+h, center_start_x:center_end_x] > 0)
            
            if len(ys) == 0:
                result.error_message = "No mask pixels in center region"
                return result
            
            # Adjust x coordinates to full image
            xs = xs + center_start_x
            ys = ys + y
            
            # Find top and bottom points
            top_y = int(np.min(ys))
            bottom_y = int(np.max(ys))
            
            # Get depth values at top and bottom
            top_depth = self.get_depth_at_point(center_x, top_y, depth_frame)
            bottom_depth = self.get_depth_at_point(center_x, bottom_y, depth_frame)
            
            if top_depth == 0 or bottom_depth == 0:
                result.error_message = "Invalid depth values"
                return result
            
            # Deproject to 3D
            top_3d = self.deproject_pixel_to_3d(center_x, top_y, top_depth)
            bottom_3d = self.deproject_pixel_to_3d(center_x, bottom_y, bottom_depth)
            
            # Calculate distance using only X and Y coordinates (ignore Z/depth)
            # This gives true vertical height independent of depth variation
            height_mm = np.sqrt(
                (top_3d[0] - bottom_3d[0])**2 +
                (top_3d[1] - bottom_3d[1])**2
            )
            
            # OLD: 3D euclidean distance including Z coordinate
            # height_mm_3d = np.sqrt(
            #     (top_3d[0] - bottom_3d[0])**2 +
            #     (top_3d[1] - bottom_3d[1])**2 +
            #     (top_3d[2] - bottom_3d[2])**2
            # )
            
            result.height_mm = height_mm
            result.center_3d = (center_x, center_y, (top_depth + bottom_depth) / 2)
            result.confidence = 0.95
            
            logger.info(f"Height measurement: {height_mm:.1f}mm")
            return result
        
        except Exception as e:
            result.error_message = f"Error measuring height: {e}"
            logger.error(result.error_message)
            return result

def ndarray_to_qpixmap(img: np.ndarray, is_bgr: bool = True) -> QPixmap:
    if img is None:
        return QPixmap()
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        bytes_per_line = 3 * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ClickableLabel(QLabel):
    """QLabel that supports mouse drag for ROI selection"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand = None
        self.origin = QPoint()
        self.roi_callback = None
        self.setMouseTracking(True)
        
    def set_roi_callback(self, callback):
        self.roi_callback = callback
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            if not self.rubberBand:
                self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()))
            self.rubberBand.show()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.rubberBand and self.rubberBand.isVisible():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.rubberBand:
            rect = self.rubberBand.geometry()
            if self.roi_callback and rect.width() > 5 and rect.height() > 5:
                # Convert widget coordinates to image coordinates
                pixmap = self.pixmap()
                if pixmap and not pixmap.isNull():
                    # Scale factor from displayed label to actual image
                    scale_x = pixmap.width() / self.width()
                    scale_y = pixmap.height() / self.height()
                    
                    img_rect = QRect(
                        int(rect.x() * scale_x),
                        int(rect.y() * scale_y),
                        int(rect.width() * scale_x),
                        int(rect.height() * scale_y)
                    )
                    self.roi_callback(img_rect)
            self.rubberBand.hide()


class RealSenseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense YOLOv11 Measurement")
        self.resize(1280, 800)

        self.streamer: Optional[RealSenseStreamer] = None
        self.detector: Optional[YOLODetector] = None
        self.depth_processor: Optional[DepthProcessor] = None
        self.stream_thread: Optional[threading.Thread] = None

        self.current_color_frame: Optional[np.ndarray] = None
        self.current_depth_frame: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_measurement: Optional[MeasurementResult] = None

        self.conf_threshold: float = 0.5
        self.last_detect_ts = 0.0
        self.detect_interval_s = 0.2  # 5 Hz detection to keep UI smooth

        self.capture_dir = (Path(__file__).parent / "captures").resolve()
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        
        self.selected_roi: Optional[QRect] = None
        self.roi_depth_stats: Optional[Dict] = None

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(50)  # 20 FPS UI refresh
        self.timer.timeout.connect(self.update_ui)

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        # Controls
        btn_start = QPushButton("Start Streaming")
        btn_stop = QPushButton("Stop Streaming")
        btn_measure = QPushButton("Take Measurement")
        btn_capture = QPushButton("Capture")

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(int(self.conf_threshold * 100))
        self.slider_conf.setToolTip("Detection confidence threshold (%)")

        self.lbl_conf_value = QLabel(f"{self.conf_threshold:.2f}")

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(btn_start)
        ctrl_layout.addWidget(btn_stop)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("Confidence:"))
        ctrl_layout.addWidget(self.slider_conf)
        ctrl_layout.addWidget(self.lbl_conf_value)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(btn_measure)
        ctrl_layout.addWidget(btn_capture)

        # Displays
        self.lbl_rgb = ClickableLabel()
        self.lbl_rgb.setText("RGB will appear here\n(Click and drag to select ROI)")
        self.lbl_rgb.setAlignment(Qt.AlignCenter)
        self.lbl_rgb.setMinimumHeight(300)
        self.lbl_rgb.setStyleSheet("background:#222; color:#bbb;")
        self.lbl_rgb.setScaledContents(False)
        self.lbl_rgb.set_roi_callback(self.on_roi_selected)

        self.lbl_depth = QLabel("Depth will appear here")
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setMinimumHeight(300)
        self.lbl_depth.setStyleSheet("background:#222; color:#bbb;")

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.lbl_rgb, 1)
        img_layout.addWidget(self.lbl_depth, 1)

        # Results area split into measurement and depth stats
        results_layout = QHBoxLayout()
        
        # Measurement results
        meas_group = QGroupBox("Measurement Results")
        meas_layout = QVBoxLayout()
        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setMinimumHeight(100)
        meas_layout.addWidget(self.txt_results)
        meas_group.setLayout(meas_layout)
        
        # Depth stats for selected ROI
        depth_group = QGroupBox("Depth Statistics (Selected ROI)")
        depth_layout = QVBoxLayout()
        self.txt_depth_stats = QTextEdit()
        self.txt_depth_stats.setReadOnly(True)
        self.txt_depth_stats.setMinimumHeight(100)
        self.txt_depth_stats.setPlainText("Select an ROI on RGB view to see depth stats")
        depth_layout.addWidget(self.txt_depth_stats)
        depth_group.setLayout(depth_layout)
        
        results_layout.addWidget(meas_group, 1)
        results_layout.addWidget(depth_group, 1)

        layout = QVBoxLayout()
        layout.addLayout(ctrl_layout)
        layout.addLayout(img_layout)
        layout.addLayout(results_layout)
        central.setLayout(layout)

        # Status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self.status.showMessage("Streaming Inactive")

        # Signals
        btn_start.clicked.connect(self.on_start)
        btn_stop.clicked.connect(self.on_stop)
        btn_measure.clicked.connect(self.on_measure)
        btn_capture.clicked.connect(self.on_capture)
        self.slider_conf.valueChanged.connect(self.on_conf_changed)

    def on_conf_changed(self, v: int):
        self.conf_threshold = max(0.0, min(1.0, v / 100.0))
        self.lbl_conf_value.setText(f"{self.conf_threshold:.2f}")
    
    def on_roi_selected(self, rect: QRect):
        """Called when user drags ROI on RGB view"""
        if self.current_depth_frame is None:
            return
        
        try:
            self.selected_roi = rect
            
            # Extract ROI from depth frame
            h, w = self.current_depth_frame.shape
            x1 = max(0, min(rect.x(), w - 1))
            y1 = max(0, min(rect.y(), h - 1))
            x2 = max(0, min(rect.x() + rect.width(), w))
            y2 = max(0, min(rect.y() + rect.height(), h))
            
            if x2 <= x1 or y2 <= y1:
                return
            
            roi_depth = self.current_depth_frame[y1:y2, x1:x2]
            
            # Filter valid depth values
            valid_depth = roi_depth[(roi_depth > 0) & (roi_depth < 65535)]
            
            if len(valid_depth) == 0:
                self.txt_depth_stats.setPlainText("No valid depth values in selected ROI")
                return
            
            # Convert to meters using depth scale
            depth_scale = self.streamer.depth_scale if self.streamer else 0.001
            valid_depth_m = valid_depth.astype(np.float32) * depth_scale
            
            # Compute statistics
            stats = {
                "min_m": float(np.min(valid_depth_m)),
                "max_m": float(np.max(valid_depth_m)),
                "mean_m": float(np.mean(valid_depth_m)),
                "median_m": float(np.median(valid_depth_m)),
                "std_m": float(np.std(valid_depth_m)),
                "valid_pixels": len(valid_depth),
                "total_pixels": roi_depth.size,
                "roi_rect": (x1, y1, x2 - x1, y2 - y1)
            }
            
            self.roi_depth_stats = stats
            
            # Display stats
            lines = [
                f"ROI: ({x1}, {y1}) - ({x2}, {y2})",
                f"Size: {x2-x1} x {y2-y1} pixels",
                f"Valid pixels: {stats['valid_pixels']} / {stats['total_pixels']}",
                "",
                f"Min distance:    {stats['min_m']:.3f} m  ({stats['min_m']*1000:.1f} mm)",
                f"Max distance:    {stats['max_m']:.3f} m  ({stats['max_m']*1000:.1f} mm)",
                f"Mean distance:   {stats['mean_m']:.3f} m  ({stats['mean_m']*1000:.1f} mm)",
                f"Median distance: {stats['median_m']:.3f} m  ({stats['median_m']*1000:.1f} mm)",
                f"Std deviation:   {stats['std_m']:.3f} m  ({stats['std_m']*1000:.1f} mm)",
            ]
            self.txt_depth_stats.setPlainText("\n".join(lines))
            
        except Exception as e:
            logger.error(f"Error computing ROI depth stats: {e}", exc_info=True)
            self.txt_depth_stats.setPlainText(f"Error: {e}")

    def on_start(self):
        try:
            if self.streamer and self.streamer.is_running:
                return
            logger.info("Starting RealSense streaming...")
            self.streamer = RealSenseStreamer()
            self.detector = YOLODetector()
            self.stream_thread = threading.Thread(target=self.streamer.run, daemon=True)
            self.stream_thread.start()
            self.timer.start()
            self.status.showMessage("Streaming Active")
        except Exception as e:
            QMessageBox.critical(self, "Start Error", str(e))

    def on_stop(self):
        try:
            if self.streamer:
                self.streamer.stop_flag = True
            self.timer.stop()
            self.status.showMessage("Streaming Inactive")
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", str(e))

    def on_measure(self):
        if self.current_color_frame is None:
            QMessageBox.information(self, "Measure", "No frames available. Start streaming first.")
            return
        if self.detector is None:
            QMessageBox.information(self, "Measure", "Detector not initialized.")
            return
        try:
            # Run detection on current frame
            result = self.detector.detect(self.current_color_frame, self.conf_threshold)
            self.current_mask = result.get("mask")

            # Initialize depth processor if needed
            if (self.depth_processor is None and self.streamer and self.streamer.intrinsics):
                self.depth_processor = DepthProcessor(self.streamer.intrinsics, self.streamer.depth_scale)

            # Run measurement if mask available
            if self.current_mask is not None and self.depth_processor and self.current_depth_frame is not None:
                height_result = self.depth_processor.measure_height_from_mask(
                    self.current_mask, self.current_depth_frame
                )
                self.current_measurement = height_result
                self.render_results()
            else:
                QMessageBox.information(self, "Measure", "No object detected in current frame.")
        except Exception as e:
            QMessageBox.critical(self, "Measurement Error", str(e))

    def render_results(self):
        m = self.current_measurement
        if not m:
            self.txt_results.setPlainText("No measurement data yet")
            return
        lines = []
        if m.height_mm is not None:
            lines.append(f"Height: {m.height_mm:.2f} mm")
        if m.center_3d is not None:
            x, y, z = m.center_3d
            lines.append(f"Center 3D: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        lines.append(f"Confidence: {m.confidence:.2%}")
        if m.error_message:
            lines.append(f"Error: {m.error_message}")
        self.txt_results.setPlainText("\n".join(lines))

    def update_ui(self):
        # Consume frames captured by streaming thread
        if self.streamer and self.streamer.is_running:
            if self.streamer.color_frame is not None and self.streamer.depth_frame is not None:
                try:
                    color_image = np.asanyarray(self.streamer.color_frame.get_data())
                    depth_image = np.asanyarray(self.streamer.depth_frame.get_data())

                    self.current_color_frame = color_image
                    self.current_depth_frame = depth_image

                    # Throttled detection
                    now = time.time()
                    if self.detector and (now - self.last_detect_ts) >= self.detect_interval_s:
                        det = self.detector.detect(color_image, self.conf_threshold)
                        self.current_mask = det.get("mask")
                        self.last_detect_ts = now

                    # Render RGB with overlay
                    display_rgb = self.current_color_frame.copy()
                    if self.current_mask is not None:
                        mask_colored = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR)
                        mask_colored[:, :, 0] = 0
                        mask_colored[:, :, 2] = 0
                        display_rgb = cv2.addWeighted(display_rgb, 0.7, mask_colored, 0.3, 0)
                    self.lbl_rgb.setPixmap(ndarray_to_qpixmap(display_rgb, is_bgr=True))

                    # Render depth map with annotations
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(self.current_depth_frame, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    
                    # Overlay ROI rectangle if selected
                    if self.selected_roi:
                        x1, y1, w, h = self.roi_depth_stats['roi_rect'] if self.roi_depth_stats else (0, 0, 0, 0)
                        if w > 0 and h > 0:
                            cv2.rectangle(depth_colormap, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                            # Annotate center with median distance
                            if self.roi_depth_stats:
                                cx = x1 + w // 2
                                cy = y1 + h // 2
                                dist_text = f"{self.roi_depth_stats['median_m']:.2f}m"
                                cv2.putText(depth_colormap, dist_text, (cx - 30, cy - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add distance scale legend
                    h_depth, w_depth = depth_colormap.shape[:2]
                    scale_text = "Distance Scale (approx):"
                    cv2.putText(depth_colormap, scale_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # Sample a few points for reference
                    sample_points = [(w_depth // 4, h_depth // 2), (w_depth // 2, h_depth // 2), (3 * w_depth // 4, h_depth // 2)]
                    depth_scale = self.streamer.depth_scale if self.streamer else 0.001
                    for i, (sx, sy) in enumerate(sample_points):
                        if 0 <= sy < h_depth and 0 <= sx < w_depth:
                            sample_depth = self.current_depth_frame[sy, sx]
                            if sample_depth > 0:
                                sample_m = sample_depth * depth_scale
                                cv2.circle(depth_colormap, (sx, sy), 3, (255, 255, 255), -1)
                                cv2.putText(depth_colormap, f"{sample_m:.2f}m", (sx - 25, sy + 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    self.lbl_depth.setPixmap(ndarray_to_qpixmap(depth_colormap, is_bgr=True))

                    # Update status
                    self.status.showMessage("Streaming Active")

                except Exception as e:
                    logger.error(f"UI update error: {e}")
        else:
            self.status.showMessage("Streaming Inactive")

    def on_capture(self):
        try:
            if self.current_color_frame is None or self.current_depth_frame is None:
                QMessageBox.information(self, "Capture", "No frames available to capture.")
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            color_path = self.capture_dir / f"color_{ts}.png"
            depth_path = self.capture_dir / f"depth_{ts}.png"
            meta_path = self.capture_dir / f"meta_{ts}.json"

            # Deep copies to avoid frame mutation during save
            color_bgr = self.current_color_frame.copy()
            depth_u16 = self.current_depth_frame.copy()

            if depth_u16.dtype != np.uint16:
                # Ensure 16-bit depth for lossless storage if possible
                depth_u16 = depth_u16.astype(np.uint16, copy=False)

            # Save images
            ok_color = cv2.imwrite(str(color_path), color_bgr)
            ok_depth = cv2.imwrite(str(depth_path), depth_u16)

            # Minimal metadata
            meta = {
                "timestamp": ts,
                "color_path": str(color_path.name),
                "depth_path": str(depth_path.name),
                "intrinsics": None,
                "depth_scale": None,
            }
            if self.streamer and self.streamer.intrinsics:
                intr = self.streamer.intrinsics
                meta["intrinsics"] = {
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "cx": intr.cx,
                    "cy": intr.cy,
                    "width": intr.width,
                    "height": intr.height,
                }
            if self.streamer:
                meta["depth_scale"] = float(self.streamer.depth_scale)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if ok_color and ok_depth:
                self.status.showMessage(f"Captured: {color_path.name}, {depth_path.name}")
            else:
                QMessageBox.warning(self, "Capture", "Failed to save one or more files.")
        except Exception as e:
            QMessageBox.critical(self, "Capture Error", str(e))


def main():
    app = QApplication([])
    win = RealSenseApp()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
