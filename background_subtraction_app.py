"""
Background Subtraction Application
Supports Uploaded images, RealSense camera, and Top Camera (HTTP stream)
"""

import cv2
import numpy as np
import sys
import threading
from pathlib import Path
from datetime import datetime
import logging
import yaml

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
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
    QFileDialog,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QFormLayout,
)

# Import custom classes
from realsense_streamer import RealSenseStreamer, HttpStreamer, DepthProcessor
from background_subtraction import BackgroundSubtraction, get_background_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        config_file = Path(config_path)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config if config else {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def ndarray_to_qpixmap(img: np.ndarray, is_bgr: bool = True) -> QPixmap:
    """Convert numpy array to QPixmap"""
    if img is None or img.size == 0:
        return QPixmap()
    if is_bgr and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        bytes_per_line = 3 * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class BackgroundSubtractionApp(QMainWindow):
    """Main application for background subtraction"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Subtraction - Uploaded vs RealSense vs Top Camera")
        self.resize(1400, 900)
        
        # Load configuration
        self.config = load_config()
        
        # Data storage
        self.background_image = None
        self.target_image = None
        self.captured_depth_frame = None  # Store depth frame when capturing from camera
        self.current_mode = "uploaded"  # "uploaded", "realsense", or "top_camera"
        
        # Streamers
        self.streamer = None
        self.stream_thread = None
        self.current_color_frame = None
        self.current_depth_frame = None
        
        # Processing Class
        self.bg_subtractor = BackgroundSubtraction()
        
        # Parameters from config or defaults
        processing_params = self.config.get('processing', {})
        self.min_deviation = processing_params.get('min_deviation', 10)
        self.max_deviation = processing_params.get('max_deviation', 255)
        self.min_contour_area = processing_params.get('min_contour_area', 1000)
        
        # Capture directory
        self.capture_dir = (Path(__file__).parent / "captures").resolve()
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        
        self._build_ui()
        
        # Timer for Camera updates
        self.timer = QTimer(self)
        ui_fps = self.config.get('ui', {}).get('update_fps', 20)
        self.timer.setInterval(int(1000 / ui_fps))  # Convert FPS to ms interval
        self.timer.timeout.connect(self.update_camera_ui)
    def _build_ui(self):
        """Build the user interface"""
        central = QWidget(self)
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout()
        
        # Mode selection
        mode_group = QGroupBox("Image Source")
        mode_layout = QHBoxLayout()
        
        self.radio_uploaded = QRadioButton("Uploaded Images")
        self.radio_realsense = QRadioButton("RealSense Camera")
        self.radio_top_cam = QRadioButton("Top Camera (HTTP)")
        self.radio_uploaded.setChecked(True)
        
        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_uploaded)
        self.btn_group.addButton(self.radio_realsense)
        self.btn_group.addButton(self.radio_top_cam)
        
        mode_layout.addWidget(self.radio_uploaded)
        mode_layout.addWidget(self.radio_realsense)
        mode_layout.addWidget(self.radio_top_cam)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        # Background controls (shared by both modes)
        bg_group = QGroupBox("Background Image (Shared)")
        bg_layout = QHBoxLayout()
        self.btn_load_bg = QPushButton("Load Background Image")
        self.btn_capture_bg = QPushButton("Capture Background")
        bg_layout.addWidget(self.btn_load_bg)
        bg_layout.addWidget(self.btn_capture_bg)
        bg_layout.addStretch()
        bg_group.setLayout(bg_layout)
        
        # Target image controls
        target_group = QGroupBox("Target Image")
        target_layout = QHBoxLayout()
        self.btn_load_target = QPushButton("Load from File")
        self.btn_start_camera = QPushButton("Start Camera")
        self.btn_stop_camera = QPushButton("Stop Camera")
        self.btn_capture_target = QPushButton("Capture from Camera")
        target_layout.addWidget(self.btn_load_target)
        target_layout.addSpacing(20)
        target_layout.addWidget(self.btn_start_camera)
        target_layout.addWidget(self.btn_stop_camera)
        target_layout.addWidget(self.btn_capture_target)
        target_layout.addStretch()
        target_group.setLayout(target_layout)
        
        controls_layout.addWidget(bg_group)
        controls_layout.addWidget(target_group)
        
        # Process button
        process_layout = QHBoxLayout()
        self.btn_process = QPushButton("Run Background Subtraction")
        self.btn_detect_rim = QPushButton("Detect Rim")
        self.btn_process.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        self.btn_detect_rim.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2196F3; color: white;")
        
        process_layout.addWidget(self.btn_process)
        process_layout.addWidget(self.btn_detect_rim)
        process_layout.addStretch()
        
        # Parameters
        params_group = QGroupBox("Background Subtraction Parameters")
        params_layout = QFormLayout()
        
        self.slider_min_dev = QSlider(Qt.Horizontal)
        self.slider_min_dev.setRange(0, 100)
        self.slider_min_dev.setValue(self.min_deviation)
        self.lbl_min_dev = QLabel(str(self.min_deviation))
        
        self.slider_max_dev = QSlider(Qt.Horizontal)
        self.slider_max_dev.setRange(0, 255)
        self.slider_max_dev.setValue(self.max_deviation)
        self.lbl_max_dev = QLabel(str(self.max_deviation))
        
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 100000)
        self.spin_min_area.setValue(self.min_contour_area)
        self.spin_min_area.setSingleStep(100)
        
        min_dev_layout = QHBoxLayout()
        min_dev_layout.addWidget(self.slider_min_dev)
        min_dev_layout.addWidget(self.lbl_min_dev)
        
        max_dev_layout = QHBoxLayout()
        max_dev_layout.addWidget(self.slider_max_dev)
        max_dev_layout.addWidget(self.lbl_max_dev)
        
        params_layout.addRow("Min Deviation:", min_dev_layout)
        params_layout.addRow("Max Deviation:", max_dev_layout)
        params_layout.addRow("Min Contour Area:", self.spin_min_area)
        
        params_group.setLayout(params_layout)
        
        # Image displays
        display_layout = QHBoxLayout()
        
        # Background
        bg_group = QGroupBox("Background Image")
        bg_layout = QVBoxLayout()
        self.lbl_background = QLabel("No background loaded")
        self.lbl_background.setAlignment(Qt.AlignCenter)
        self.lbl_background.setMinimumHeight(250)
        self.lbl_background.setStyleSheet("background:#222; color:#bbb;")
        bg_layout.addWidget(self.lbl_background)
        bg_group.setLayout(bg_layout)
        
        # Target
        target_group = QGroupBox("Target Image")
        target_layout = QVBoxLayout()
        self.lbl_target = QLabel("No target loaded")
        self.lbl_target.setAlignment(Qt.AlignCenter)
        self.lbl_target.setMinimumHeight(250)
        self.lbl_target.setStyleSheet("background:#222; color:#bbb;")
        target_layout.addWidget(self.lbl_target)
        target_group.setLayout(target_layout)
        
        display_layout.addWidget(bg_group)
        display_layout.addWidget(target_group)
        
        # Results display
        results_layout = QHBoxLayout()
        
        # Difference heatmap
        diff_group = QGroupBox("Difference Heatmap")
        diff_layout = QVBoxLayout()
        self.lbl_heatmap = QLabel("Process images to see heatmap")
        self.lbl_heatmap.setAlignment(Qt.AlignCenter)
        self.lbl_heatmap.setMinimumHeight(250)
        self.lbl_heatmap.setStyleSheet("background:#222; color:#bbb;")
        diff_layout.addWidget(self.lbl_heatmap)
        diff_group.setLayout(diff_layout)
        
        # Mask result
        mask_group = QGroupBox("Background Mask (Result)")
        mask_layout = QVBoxLayout()
        self.lbl_mask = QLabel("Process images to see mask")
        self.lbl_mask.setAlignment(Qt.AlignCenter)
        self.lbl_mask.setMinimumHeight(250)
        self.lbl_mask.setStyleSheet("background:#222; color:#bbb;")
        mask_layout.addWidget(self.lbl_mask)
        mask_group.setLayout(mask_layout)
        
        # Overlay result
        overlay_group = QGroupBox("Overlay Result")
        overlay_layout = QVBoxLayout()
        self.lbl_overlay = QLabel("Process images to see overlay")
        self.lbl_overlay.setAlignment(Qt.AlignCenter)
        self.lbl_overlay.setMinimumHeight(250)
        self.lbl_overlay.setStyleSheet("background:#222; color:#bbb;")
        overlay_layout.addWidget(self.lbl_overlay)
        overlay_group.setLayout(overlay_layout)
        
        results_layout.addWidget(diff_group)
        results_layout.addWidget(mask_group)
        results_layout.addWidget(overlay_group)
        
        
        # Status info
        self.lbl_status = QLabel("Ready. Load background image first (shared for both modes), then load or capture target.")
        self.lbl_status.setStyleSheet("font-size: 12px; padding: 5px; background:#333; color:#0f0;")
        
        # Assemble main layout
        main_layout.addWidget(mode_group)
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(process_layout)
        main_layout.addWidget(params_group)
        main_layout.addLayout(display_layout)
        main_layout.addLayout(results_layout)
        main_layout.addWidget(self.lbl_status)
        
        central.setLayout(main_layout)
        
        # Connect signals
        self.radio_uploaded.toggled.connect(self.on_mode_changed)
        self.radio_realsense.toggled.connect(self.on_mode_changed)
        self.radio_top_cam.toggled.connect(self.on_mode_changed)
        
        self.btn_load_bg.clicked.connect(self.load_background)
        self.btn_capture_bg.clicked.connect(self.capture_background)
        self.btn_load_target.clicked.connect(self.load_target)
        
        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_capture_target.clicked.connect(self.capture_target)
        
        self.btn_process.clicked.connect(self.process_background_subtraction)
        self.btn_detect_rim.clicked.connect(self.detect_rim_from_mask)
        
        self.slider_min_dev.valueChanged.connect(self.update_params)
        self.slider_max_dev.valueChanged.connect(self.update_params)
        self.spin_min_area.valueChanged.connect(self.update_params)
        
        # Store result images for saving
        self.result_heatmap = None
        self.result_mask = None
        self.bg_mask = None
        self.result_overlay = None
        self.wheel_center_info = None
    
    def on_mode_changed(self):
        """Handle mode selection change"""
        if self.radio_uploaded.isChecked():
            self.current_mode = "uploaded"
            self.lbl_status.setText("Mode: Uploaded - Load shared background, then load target from file")
            self.stop_camera() # Stop camera if we switch to uploaded
            
        elif self.radio_realsense.isChecked():
            self.current_mode = "realsense"
            self.lbl_status.setText("Mode: RealSense - Load shared background, start camera and capture target")
            self.stop_camera() # Stop any previous streamer
            
        elif self.radio_top_cam.isChecked():
            self.current_mode = "top_camera"
            self.lbl_status.setText("Mode: Top Camera - Load shared background, start camera and capture target")
            self.stop_camera() # Stop any previous streamer
    
    def update_params(self):
        """Update parameters from UI"""
        self.min_deviation = self.slider_min_dev.value()
        self.max_deviation = self.slider_max_dev.value()
        self.min_contour_area = self.spin_min_area.value()
        
        self.lbl_min_dev.setText(str(self.min_deviation))
        self.lbl_max_dev.setText(str(self.max_deviation))
        
        # Update subtractor parameters
        self.bg_subtractor.lower_threshold = self.min_deviation
        self.bg_subtractor.upper_threshold = self.max_deviation
    
    def load_background(self):
        """Load background image from file (used for both modes)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.background_image = img
                self.lbl_background.setPixmap(
                    ndarray_to_qpixmap(img, is_bgr=True).scaled(
                        self.lbl_background.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
                self.lbl_status.setText(f"Background loaded (shared): {Path(file_path).name}")
                logger.info(f"Loaded background from file: {file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")
    
    def load_target(self):
        """Load target image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.target_image = img
                self.lbl_target.setPixmap(
                    ndarray_to_qpixmap(img, is_bgr=True).scaled(
                        self.lbl_target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
                self.lbl_status.setText(f"Target loaded: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")

    def capture_background(self):
        """Capture current frame as background (for RealSense or HTTP modes)"""
        if self.current_color_frame is None:
            QMessageBox.warning(self, "Capture", "No camera frame available. Start camera first.")
            return

        self.background_image = self.current_color_frame.copy()

        self.lbl_background.setPixmap(
            ndarray_to_qpixmap(self.background_image, is_bgr=True).scaled(
                self.lbl_background.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.lbl_status.setText("Background captured from camera")

    def start_camera(self):
        """Start the appropriate camera based on current mode"""
        if self.streamer and self.streamer.is_running:
            self.lbl_status.setText("Camera already running")
            return
        
        try:
            if self.current_mode == "realsense":
                self.streamer = RealSenseStreamer()
            elif self.current_mode == "top_camera":
                # Get HTTP URL from config
                http_config = self.config.get('http_streamer', {})
                http_url = http_config.get('url', "http://192.168.1.100:8080/video_feed")
                http_timeout = http_config.get('timeout', 5)
                http_pixel_to_mm = http_config.get('pixel_to_mm', 1.0)
                
                self.streamer = HttpStreamer(url=http_url, timeout=http_timeout, pixel_to_mm=http_pixel_to_mm)
                logger.info(f"Initialized HTTP streamer with URL: {http_url}")
            else:
                 QMessageBox.warning(self, "Mode Error", "Please select RealSense or Top Camera mode first")
                 return
            
            self.stream_thread = threading.Thread(target=self.streamer.run, daemon=True)
            self.stream_thread.start()
            self.timer.start()
            self.lbl_status.setText(f"{self.current_mode.capitalize()} started - Waiting for frames...")
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {e}")
            logger.error(f"Camera error: {e}", exc_info=True)
    
    def stop_camera(self):
        """Stop current camera stream"""
        if self.streamer:
            self.streamer.stop_flag = True
            # Wait for thread? Not strictly necessary if we just clear refs
        
        self.timer.stop()
        self.lbl_status.setText("Camera stopped")
        
        # Cleanup handled in streamer's run loop mostly, but we can ensure
        # self.streamer = None # Keep ref for cleanup if needed or reuse? 
        # Better to re-instantiate for clean start
        
    def update_camera_ui(self):
        """Update UI with frames from current streamer"""
        if not self.streamer or not self.streamer.is_running:
            return
        
        try:
            # Check frame availability
            if self.streamer.color_frame is not None:
                # Handle RealSense data (pyrealsense2 frame object)
                if self.current_mode == "realsense":
                    # Check both frames
                     if self.streamer.depth_frame:
                        color_data = np.asanyarray(self.streamer.color_frame.get_data())
                        depth_data = np.asanyarray(self.streamer.depth_frame.get_data())
                        
                        self.current_color_frame = color_data.copy()
                        self.current_depth_frame = depth_data.copy()
                
                # Handle Top Camera (HTTP) data (numpy array)
                elif self.current_mode == "top_camera":
                     self.current_color_frame = self.streamer.color_frame.copy()
                     self.current_depth_frame = None # No depth
                
                # Update UI
                if self.current_color_frame is not None:
                    # Show current frame in target view if no static target loaded (or just always update for live view?)
                    # Original logic was: if self.target_image is None
                    # But for streaming we usually want to see the stream
                    # Let's stick to original logic: if target image is NOT set (captured), show stream.
                    # Once captured, it shows captured image.
                    if self.target_image is None:
                        self.lbl_target.setPixmap(
                            ndarray_to_qpixmap(self.current_color_frame, is_bgr=True).scaled(
                                self.lbl_target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                            )
                        )
                        
        except Exception as e:
            logger.error(f"UI update error: {e}")
    
    def capture_target(self):
        """Capture current frame as target"""
        if self.current_color_frame is None:
            QMessageBox.warning(self, "Capture", "No camera frame available. Start camera first.")
            return
        
        self.target_image = self.current_color_frame.copy()
        
        # Handle depth capture
        if self.current_depth_frame is not None:
            self.captured_depth_frame = self.current_depth_frame.copy()
            logger.info(f"Captured depth frame with shape: {self.captured_depth_frame.shape}")
        else:
            self.captured_depth_frame = None
            logger.warning("No depth frame available (or not supported in this mode)")
        
        self.lbl_target.setPixmap(
            ndarray_to_qpixmap(self.target_image, is_bgr=True).scaled(
                self.lbl_target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        
        msg = "Target captured"
        if self.captured_depth_frame is not None:
            msg += " (with depth)"
        else:
            msg += " (RGB only)"
        self.lbl_status.setText(msg)
    
    def find_wheel_center_and_depth(self, mask: np.ndarray, depth_frame: np.ndarray = None) -> dict:
        """Find wheel center from mask using bottom 75% of contour
        
        Args:
            mask: Binary mask of detected wheel
            depth_frame: Optional depth frame for depth measurement
            
        Returns:
            Dictionary with center coordinates and depth info
        """
        result = {
            'center_x': None,
            'center_y': None,
            'depth_mm': None,
            'depth_m': None,
            'error': None
        }
        
        try:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                result['error'] = "No contours found in mask"
                return result
            
            # Get largest contour (main wheel)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Use bottom 75% of the contour to avoid fins at top
            bottom_75_y = y + int(h * 0.25)  # Start from 25% down
            
            # Filter contour points to only include bottom 75%
            bottom_points = []
            for point in main_contour:
                px, py = point[0]
                if py >= bottom_75_y:
                    bottom_points.append(point)
            
            if len(bottom_points) < 5:
                # Fallback to full contour if not enough points
                bottom_points = main_contour
                logger.warning("Not enough points in bottom 75%, using full contour")
            
            bottom_contour = np.array(bottom_points)
            
            # Calculate center of bottom 75% region
            M = cv2.moments(bottom_contour)
            
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                # Fallback to bounding box center
                center_x = x + w // 2
                center_y = y + int(h * 0.625)  # Center of bottom 75%
            
            result['center_x'] = center_x
            result['center_y'] = center_y
            
            # Get depth at center if depth frame available
            if depth_frame is not None:
                depth_value = self.get_depth_at_point(center_x, center_y, depth_frame)
                
                if depth_value > 0:
                    # Convert to millimeters (assuming depth_frame is in mm or needs conversion)
                    # For RealSense, depth is typically in mm when using rs.format.z16
                    result['depth_mm'] = float(depth_value)
                    result['depth_m'] = float(depth_value) / 1000.0
                else:
                    result['error'] = "Invalid depth at wheel center"
            else:
                result['error'] = "Depth frame not available"
            
            return result
            
        except Exception as e:
            result['error'] = f"Error finding wheel center: {e}"
            logger.error(result['error'], exc_info=True)
            return result
    
    def get_depth_at_point(self, x: int, y: int, depth_frame: np.ndarray, window_size: int = 30) -> float:
        """Get median depth value in a window around specified point
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            depth_frame: Depth image array
            window_size: Size of window for median calculation (default 30x30)
            
        Returns:
            Depth value in same units as depth_frame
        """
        h, w = depth_frame.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Try progressively larger windows if no valid depth found
        window_sizes = [window_size, 50, 100, 150]
        
        for ws in window_sizes:
            half_window = ws // 2
            x_start = max(x - half_window, 0)
            x_end = min(x + half_window + 1, w)
            y_start = max(y - half_window, 0)
            y_end = min(y + half_window + 1, h)
            
            window = depth_frame[y_start:y_end, x_start:x_end]
            valid_depths = window[(window > 0) & (window < 65535)]
            
            if len(valid_depths) > 0:
                median_depth = float(np.median(valid_depths))
                if ws > window_size:
                    logger.info(f"Found valid depth {median_depth:.1f} at ({x}, {y}) using expanded window {ws}x{ws} (found {len(valid_depths)} valid pixels)")
                else:
                    logger.debug(f"Found valid depth {median_depth:.1f} at ({x}, {y}) using window {ws}x{ws} (found {len(valid_depths)} valid pixels)")
                return median_depth
        
        logger.warning(f"No valid depth found at ({x}, {y}) even with 150x150 window")
        return 0.0
    
    def detect_rim_from_mask(self):
        """Detect rim from the background subtraction mask"""
        if self.bg_mask is None:
            QMessageBox.warning(self, "Rim Detection", "Run background subtraction first.")
            return
        
        if self.wheel_center_info is None or self.wheel_center_info.get('center_x') is None:
            QMessageBox.warning(self, "Rim Detection", "Wheel center not detected. Process background subtraction first.")
            return
        
        # Use captured depth frame if available, otherwise current depth frame
        depth_frame = None
        if self.current_mode == "realsense":
            if self.captured_depth_frame is not None:
                depth_frame = self.captured_depth_frame
                logger.info("Using captured depth frame for rim detection")
            else:
                depth_frame = self.current_depth_frame
                logger.info("Using current depth frame for rim detection")
        
        try:
            self.lbl_status.setText("Detecting rim...")
            
            # Get parameters
            depth_tolerance_pct = self.spin_depth_tol.value() / 100.0
            use_depth_filter = self.chk_depth_filter.isChecked()
            
            # Get wheel center depth
            center_depth_mm = self.wheel_center_info.get('depth_mm')
            # If no depth frame, force depth filter off
            if depth_frame is None:
                use_depth_filter = False
                center_depth_mm = None
            # If depth filter requested but depth is invalid, disable filtering
            if use_depth_filter and (center_depth_mm is None or center_depth_mm <= 0):
                logger.warning("Wheel center depth unavailable; disabling depth filter.")
                use_depth_filter = False
            
            # Apply depth filter to mask - VECTORIZED VERSION for better performance
            # Convert depth frame to float32 for better precision (only if we have depth)
            depth_frame_float = depth_frame.astype(np.float32) if depth_frame is not None else None
            
            actual_min_depth = None
            actual_max_depth = None
            actual_median_depth = None
            depth_min = None
            depth_max = None
            
            if use_depth_filter and depth_frame_float is not None:
                # Analyze depth range in the current mask and build a filtered mask around center depth
                mask_depths = depth_frame_float[self.bg_mask > 0]
                valid_mask_depths = mask_depths[mask_depths > 0]
                
                if len(valid_mask_depths) == 0:
                    logger.warning("No valid depth pixels in mask; skipping depth filter")
                    use_depth_filter = False
                else:
                    actual_min_depth = float(np.min(valid_mask_depths))
                    actual_max_depth = float(np.max(valid_mask_depths))
                    actual_median_depth = float(np.median(valid_mask_depths))
                    # Use ABSOLUTE mm tolerance (constant across all distances)
                    depth_min = center_depth_mm - self.depth_tolerance_mm
                    depth_max = center_depth_mm + self.depth_tolerance_mm
                    logger.info(f"Depth filter ON (ABSOLUTE): center {center_depth_mm:.1f}mm, range {depth_min:.1f}-{depth_max:.1f}mm (±{self.depth_tolerance_mm}mm), mask depth {actual_min_depth:.1f}-{actual_max_depth:.1f}mm")
            
            if use_depth_filter and depth_frame_float is not None:
                depth_condition = (depth_frame_float >= depth_min) & (depth_frame_float <= depth_max)
                mask_condition = self.bg_mask > 0
                depth_filtered_mask = np.where(depth_condition & mask_condition, 255, 0).astype(np.uint8)
                
                # Apply morphological operations to fill gaps and smooth edges
                # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                # depth_filtered_mask = cv2.morphologyEx(depth_filtered_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
                
                # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # depth_filtered_mask = cv2.morphologyEx(depth_filtered_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            else:
                # Use the background subtraction mask directly to compare behavior without depth gating
                depth_filtered_mask = self.bg_mask.copy()
                logger.info("Depth filter OFF: running Hough on background mask only")
            
            # Find contours in depth-filtered mask
            contours, _ = cv2.findContours(depth_filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found after depth filtering")
                return
            
            # IMPROVED HOUGH LINE DETECTION: Focus on rim band only
            wheel_center_y = self.wheel_center_info['center_y']
            wheel_center_x = self.wheel_center_info['center_x']
            
            # Get wheel dimensions from largest contour for adaptive scaling
            main_contour = max(contours, key=cv2.contourArea)
            wheel_x, wheel_y, wheel_w, wheel_h = cv2.boundingRect(main_contour)
            logger.info(f"Wheel bounding box: x={wheel_x}, y={wheel_y}, w={wheel_w}, h={wheel_h}")
            
            # Create focused region: horizontal band around wheel center (ADAPTIVE based on wheel height)
            h, w = depth_filtered_mask.shape
            band_height = int(wheel_h * (1))  # % of actual wheel height
            y_band_start = max(0, wheel_center_y - band_height)
            y_band_end = min(h, wheel_center_y + band_height)
            logger.info(f"Adaptive band: height={band_height}px ({self.band_height_percent}% of wheel {wheel_h}px)")
            
            # Extract rim band only
            rim_band = depth_filtered_mask[y_band_start:y_band_end, :].copy()
            
            # Dilate the mask to smooth sharp edges and connect gaps before edge detection
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            rim_band_dilated = cv2.dilate(rim_band, dilate_kernel, iterations=1)
            
           
            if self.target_image is not None:
                hough_viz = self.target_image.copy()
                
            
            
        
            if 1:
                # Fallback: use projection method on rim band
                vertical_proj = np.sum(rim_band, axis=1)
                non_zero_rows = np.where(vertical_proj > vertical_proj.max() * 0.8)[0]
                if len(non_zero_rows) > 0:
                    y_top = non_zero_rows[0] + y_band_start
                    y_bottom = non_zero_rows[-1] + y_band_start
                else:
                    y_top = y_band_start + 10
                    y_bottom = y_band_end - 10
                logger.info(f"Projection fallback: y={y_top} to y={y_bottom}")
            
            # Find left and right boundaries from mask width (not from vertical lines)
            # Use horizontal projection to find leftmost and rightmost non-zero columns
            horizontal_proj = np.sum(depth_filtered_mask, axis=0)
            non_zero_cols = np.where(horizontal_proj > 0)[0]
            if len(non_zero_cols) > 0:
                x_left = non_zero_cols[0]
                x_right = non_zero_cols[-1]
                logger.info(f"Left/Right boundaries from mask width: x={x_left} to x={x_right}")
            else:
                x_left = max(0, wheel_center_x - 150)
                x_right = min(depth_filtered_mask.shape[1] - 1, wheel_center_x + 150)
                logger.info(f"Fallback left/right: x={x_left} to x={x_right}")
            
            # Add margin for safety
            margin = 0
            y_top = max(0, y_top - margin)
            y_bottom = min(depth_filtered_mask.shape[0] - 1, y_bottom + margin)
            x_left = max(0, x_left - margin)
            x_right = min(depth_filtered_mask.shape[1] - 1, x_right + margin)
            
            # Draw final chosen boundaries (top and bottom only) in WHITE
            cv2.line(hough_viz, (0, y_top), (w-1, y_top), (255, 255, 255), 3)  # Top boundary
            cv2.line(hough_viz, (0, y_bottom), (w-1, y_bottom), (255, 255, 255), 3)  # Bottom boundary
            
            # Draw final rectangle in YELLOW (BGR: 0, 255, 255)
            cv2.rectangle(hough_viz, (x_left, y_top), (x_right, y_bottom), (0, 255, 255), 2)
            
            # Add legend text
            cv2.putText(hough_viz, "RED=All Lines | BLUE=Horizontal Filtered | WHITE=Top/Bottom | YELLOW=Final Rect", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display binary mask in lbl_mask (Background Mask Result)
            self.lbl_mask.setPixmap(
                ndarray_to_qpixmap(depth_filtered_mask, is_bgr=False).scaled(
                    self.lbl_mask.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            
            # Display Hough visualization with lines and rectangle in lbl_overlay
            self.lbl_overlay.setPixmap(
                ndarray_to_qpixmap(hough_viz, is_bgr=True).scaled(
                    self.lbl_overlay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            
            # Create rim mask from detected boundaries
            rim_only_mask = np.zeros_like(depth_filtered_mask)
            rim_only_mask[y_top:y_bottom+1, x_left:x_right+1] = depth_filtered_mask[y_top:y_bottom+1, x_left:x_right+1]
            
            # Create rectangular contour from boundaries
            main_contour = np.array([
                [[x_left, y_top]], 
                [[x_right, y_top]], 
                [[x_right, y_bottom]], 
                [[x_left, y_bottom]]
            ])
            
            # Use the rim mask
            depth_filtered_mask = rim_only_mask
            
            logger.info(f"Hough rim detection: Rect[{x_left},{y_top}] to [{x_right},{y_bottom}]")
            logger.info(f"Main contour dimensions: width={x_right-x_left}px, height={y_bottom-y_top}px")
            
            # Final rim mask from detected contour
            rim_mask = np.zeros_like(depth_filtered_mask)
            cv2.drawContours(rim_mask, [main_contour], -1, 255, -1)
            
            # Calculate rim properties
            rim_area = cv2.contourArea(main_contour)
            rim_x, rim_y, rim_w, rim_h = cv2.boundingRect(main_contour)
            
            logger.info(f"Rim bounding box from contour: x={rim_x}, y={rim_y}, w={rim_w}, h={rim_h}")
            
            # Get rim center
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                rim_center_x = int(M["m10"] / M["m00"])
                rim_center_y = int(M["m01"] / M["m00"])
            else:
                rim_center_x = rim_x + rim_w // 2
                rim_center_y = rim_y + rim_h // 2
            
            # Get rim depth only if a depth frame is available
            rim_depth = None
            if depth_frame is not None:
                rim_depth = self.get_depth_at_point(rim_center_x, rim_center_y, depth_frame)
            
            # Convert pixel measurements to mm using depth projection
            # Use center-based approach like measure_height_from_mask in realsense.py
            rim_height_mm = None
            measurement_points = None  # Store measurement points for visualization
            
            # Try to initialize depth processor if not already done and streamer is available
            if self.depth_processor is None and self.streamer is not None:
                if self.streamer.intrinsics and self.streamer.depth_scale:
                    self.depth_processor = DepthProcessor(self.streamer.intrinsics, self.streamer.depth_scale)
                    logger.info("Depth processor initialized on-demand with camera intrinsics")
                else:
                    logger.warning("Streamer available but intrinsics not ready")
            
            # Log camera intrinsics for debugging
            if self.depth_processor is not None:
                intr = self.depth_processor.intrinsics
                logger.info(f"Camera intrinsics: fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, cx={intr['cx']:.2f}, cy={intr['cy']:.2f}, resolution={intr['width']}x{intr['height']}")
                logger.info(f"Depth scale: {self.depth_processor.depth_scale}")
            
            if depth_frame is not None and self.depth_processor is not None and rim_depth is not None and rim_depth > 0:
                try:
                    # Use center 40% of rim mask for height calculation (similar to measure_height_from_mask)
                    rim_center_start_x = rim_x + int(rim_w * 0.30)
                    rim_center_end_x = rim_x + int(rim_w * 0.70)
                    
                    # Find actual mask pixels in center vertical band for height measurement
                    ys_vertical, xs_vertical = np.where(
                        rim_mask[rim_y:rim_y+rim_h, rim_center_start_x:rim_center_end_x] > 0
                    )
                    
                    # Calculate HEIGHT using vertical center band
                    if len(ys_vertical) > 0:
                        # Adjust coordinates to full image
                        xs_vertical_adj = xs_vertical + rim_center_start_x
                        ys_vertical_adj = ys_vertical + rim_y
                        
                        # Find top and bottom points
                        top_y_actual = int(np.min(ys_vertical_adj))
                        bottom_y_actual = int(np.max(ys_vertical_adj))
                        
                        # Use rim center x for height measurement
                        height_center_x = rim_center_x
                        
                        # Get depth at top and bottom
                        top_depth = self.depth_processor.get_depth_at_point(height_center_x, top_y_actual, depth_frame)
                        bottom_depth = self.depth_processor.get_depth_at_point(height_center_x, bottom_y_actual, depth_frame)
                        if top_depth != 0 or bottom_depth != 0:
                            if top_depth == 0:
                                top_depth = bottom_depth
                            if bottom_depth == 0:
                                bottom_depth = top_depth
                        if top_depth == 0 and bottom_depth == 0:
                            if wheel_center_x!=0 and wheel_center_y!=0:
                                center_depth = self.depth_processor.get_depth_at_point(wheel_center_x, wheel_center_y, depth_frame)
                                top_depth = center_depth
                                bottom_depth = center_depth
                        logger.info(f"Depth values: top_depth={top_depth:.2f}mm at (x={height_center_x}, y={top_y_actual}), bottom_depth={bottom_depth:.2f}mm at (x={height_center_x}, y={bottom_y_actual})")
                        
                        if top_depth > 0 and bottom_depth > 0:
                            # Deproject to 3D
                            top_3d = self.depth_processor.deproject_pixel_to_3d(height_center_x, top_y_actual, top_depth)
                            bottom_3d = self.depth_processor.deproject_pixel_to_3d(height_center_x, bottom_y_actual, bottom_depth)
                            
                            logger.info(f"3D points: top={top_3d}, bottom={bottom_3d}")
                            
                            # Calculate height using only X and Y coordinates (ignore Z/depth)
                            delta_x = top_3d[0] - bottom_3d[0]
                            delta_y = top_3d[1] - bottom_3d[1]
                            
                            logger.info(f"Deltas: delta_x={delta_x}, delta_y={delta_y}")
                            
                            rim_height_mm = np.sqrt(delta_x**2 + delta_y**2)
                            
                            # Store measurement points for visualization
                            measurement_points = {
                                'top': (height_center_x, top_y_actual),
                                'bottom': (height_center_x, bottom_y_actual)
                            }
                            
                            logger.info(f"Rim height: {rim_height_mm:.2f}mm (top_y={top_y_actual}, bottom_y={bottom_y_actual}, center_x={height_center_x})")
                        else:
                            logger.warning("Invalid depth for height measurement")
                    else:
                        logger.warning("No mask pixels in vertical center band for height measurement")
                    
                except Exception as e:
                    logger.error(f"Error converting rim measurements to mm: {e}", exc_info=True)
                    rim_height_mm = None
            else:
                logger.warning(f"Cannot compute mm measurements - depth_frame: {depth_frame is not None}, depth_processor: {self.depth_processor is not None}, rim_depth: {rim_depth}")
            
            # Store results
            self.rim_detection_result = {
                'contour': main_contour,
                'mask': rim_mask,
                'center_x': rim_center_x,
                'center_y': rim_center_y,
                'depth_mm': float(rim_depth) if rim_depth is not None else None,
                'depth_m': float(rim_depth) / 1000.0 if rim_depth is not None else None,
                'area_pixels': int(rim_area),
                'bbox': (rim_x, rim_y, rim_w, rim_h),
                'width_pixels': rim_w,
                'height_pixels': rim_h,
                'height_mm': float(rim_height_mm) if rim_height_mm is not None else None
            }
            
            # Create rim overlay
            if self.target_image is not None:
                rim_overlay = self.target_image.copy()
                
                # Draw rim contour in cyan
                cv2.drawContours(rim_overlay, [main_contour], -1, (255, 255, 0), 3)
                
                # Draw rim center
                cv2.drawMarker(rim_overlay, (rim_center_x, rim_center_y), 
                              (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
                cv2.circle(rim_overlay, (rim_center_x, rim_center_y), 5, (255, 0, 255), -1)
                
                # Draw bounding box
                cv2.rectangle(rim_overlay, (rim_x, rim_y), (rim_x + rim_w, rim_y + rim_h),
                            (0, 255, 255), 2)
                
                # Draw measurement points (top_y and bottom_y where depth is read)
                if measurement_points is not None:
                    top_pt = measurement_points['top']
                    bottom_pt = measurement_points['bottom']
                    
                    # Draw top measurement point in GREEN
                    cv2.circle(rim_overlay, top_pt, 8, (0, 255, 0), -1)
                    cv2.circle(rim_overlay, top_pt, 12, (0, 255, 0), 2)
                    cv2.putText(rim_overlay, "TOP", (top_pt[0] + 15, top_pt[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw bottom measurement point in RED
                    cv2.circle(rim_overlay, bottom_pt, 8, (0, 0, 255), -1)
                    cv2.circle(rim_overlay, bottom_pt, 12, (0, 0, 255), 2)
                    cv2.putText(rim_overlay, "BOTTOM", (bottom_pt[0] + 15, bottom_pt[1] + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw line connecting measurement points
                    cv2.line(rim_overlay, top_pt, bottom_pt, (255, 255, 255), 2)
                    
                    # Add height label if available
                    if rim_height_mm is not None:
                        mid_y = (top_pt[1] + bottom_pt[1]) // 2
                        cv2.putText(rim_overlay, f"{rim_height_mm:.1f}mm", (top_pt[0] + 20, mid_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add labels
                cv2.putText(rim_overlay, f"Rim: {rim_w}x{rim_h}px", (rim_x, rim_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                self.rim_overlay = rim_overlay
                
                # Display
                self.lbl_overlay.setPixmap(
                    ndarray_to_qpixmap(rim_overlay, is_bgr=True).scaled(
                        self.lbl_overlay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            
            # Display results
            info_lines = [
                f"Rim Center (X, Y): ({rim_center_x}, {rim_center_y}) pixels",
                f"Rim Size: {rim_w} x {rim_h} pixels",
            ]
            
            # Add height in mm if available
            if rim_height_mm is not None:
                info_lines.append(f"Rim Height (3D): {rim_height_mm:.2f} mm")
            else:
                info_lines.append(f"Rim Height (3D): Not available (need depth + intrinsics)")
            
            info_lines.append(f"Rim Area: {rim_area:.0f} pixels²")
            info_lines.append(f"Depth Filter: {'ON' if use_depth_filter else 'OFF'}")

            if center_depth_mm is not None:
                info_lines.append(f"Wheel Center Depth: {center_depth_mm:.1f} mm")
            else:
                info_lines.append("Wheel Center Depth: n/a (no depth)")
            
            # Add actual depth range analysis

            
            # Update status with height measurement if available
            if rim_height_mm is not None:
                self.lbl_status.setText(f"Rim detected: {rim_w}x{rim_h} px (height: {rim_height_mm:.2f} mm)")
            else:
                self.lbl_status.setText(f"Rim detected: {rim_w}x{rim_h} pixels")
            
            # logger.info(f"Rim detected at ({rim_center_x}, {rim_center_y}) with depth {rim_depth:.1f}mm")
            
        except Exception as e:
            QMessageBox.critical(self, "Rim Detection Error", f"Failed to detect rim: {e}")
            logger.error(f"Rim detection error: {e}", exc_info=True)

    def process_background_subtraction(self):
        """Run background subtraction algorithm"""
        background_to_use = self.background_image

        if self.current_mode in ("realsense", "top_camera"):
            if background_to_use is None:
                background_to_use = get_background_from_config(
                    self.config, self.current_mode, Path(__file__).parent
                )
            if background_to_use is None:
                QMessageBox.warning(
                    self,
                    "Process",
                    "Background image required for this mode (check config paths for this camera)."
                )
                return
        else:
            # Uploaded mode: background must be provided by user; no config fallback
            if background_to_use is None:
                QMessageBox.warning(
                    self,
                    "Process",
                    "Background image is required. Load a background before processing."
                )
                return

        if self.target_image is None:
            QMessageBox.warning(
                self, "Process", 
                "Target image is required.\n"
                "Load or capture a target before processing."
            )
            return
        
        try:
            self.lbl_status.setText("Processing background subtraction...")
            
            # Update subtractor parameters
            self.bg_subtractor.lower_threshold = self.min_deviation
            self.bg_subtractor.upper_threshold = self.max_deviation
            
            # Perform subtraction and thresholding
            heatmap_color, bg_mask = self.bg_subtractor.process(background_to_use, self.target_image)
            
            self.result_heatmap = heatmap_color
            
            # Filter by contour area (UI specific logic)
            contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filtered_mask = np.zeros_like(bg_mask)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_contour_area:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
                    valid_contours.append(contour)
            
            self.result_mask = filtered_mask
            self.bg_mask = filtered_mask # if smaller contours should be removed.
            
            # Create overlay (UI specific logic)
            # Ensure target image size matches
            target_image = self.target_image.copy()
            if target_image.shape[:2] != filtered_mask.shape:
                 target_image = cv2.resize(target_image, (filtered_mask.shape[1], filtered_mask.shape[0]))
            
            overlay = target_image.copy()
            mask_colored = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0  # Remove blue
            mask_colored[:, :, 2] = 0  # Remove red (keep only green)
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # Draw contours on overlay
            cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), 2)
            self.result_overlay = overlay
            
            # Find wheel center and depth
            # Use captured depth frame if available, otherwise current depth frame
            depth_frame = None
            if self.current_mode == "realsense":
                if self.captured_depth_frame is not None:
                    depth_frame = self.captured_depth_frame
                    logger.info("Using captured depth frame for wheel center detection")
                else:
                    depth_frame = self.current_depth_frame
                    logger.info("Using current depth frame for wheel center detection")
            elif self.current_mode == "top_camera":
                # Top camera (HTTP) does not have depth
                depth_frame = None
                logger.info("Top camera mode: No depth frame available")

                # Compute diameter in mm for top camera using streamer helper
                diameter_px, diameter_mm = (None, None)
                if isinstance(self.streamer, HttpStreamer):
                    diameter_px, diameter_mm = self.streamer.compute_diameter_mm(filtered_mask)

            
            wheel_info = self.find_wheel_center_and_depth(filtered_mask, depth_frame)
            self.wheel_center_info = wheel_info
            
            # Draw center on overlay
            if wheel_info['center_x'] is not None and wheel_info['center_y'] is not None:
                cx, cy = wheel_info['center_x'], wheel_info['center_y']
                # Draw crosshair at center
                cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
                cv2.circle(overlay, (cx, cy), 5, (0, 255, 255), -1)
                # Draw label
                label = f"Center: ({cx}, {cy})"
                cv2.putText(overlay, label, (cx + 15, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display results
            self.lbl_heatmap.setPixmap(
                ndarray_to_qpixmap(heatmap_color, is_bgr=True).scaled(
                    self.lbl_heatmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            
            self.lbl_mask.setPixmap(
                ndarray_to_qpixmap(filtered_mask, is_bgr=False).scaled(
                    self.lbl_mask.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            
            self.lbl_overlay.setPixmap(
                ndarray_to_qpixmap(overlay, is_bgr=True).scaled(
                    self.lbl_overlay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            
            # Display wheel center info
            info_lines = []
            if wheel_info['center_x'] is not None:
                info_lines.append(f"Wheel Center (X, Y): ({wheel_info['center_x']}, {wheel_info['center_y']}) pixels")
            if wheel_info['depth_m'] is not None:
                info_lines.append(f"Depth at Center: {wheel_info['depth_m']:.3f} m ({wheel_info['depth_mm']:.1f} mm)")
            elif wheel_info['depth_mm'] is None and self.current_mode == "realsense":
                info_lines.append("Depth: Not available (no depth frame)")
            elif wheel_info['depth_mm'] is None and self.current_mode == "top_camera":
                 info_lines.append("Depth: Not available (Top Camera mode)")
            elif wheel_info['depth_mm'] is None and self.current_mode == "uploaded":
                info_lines.append("Depth: Not available (uploaded image mode)")
            if self.current_mode == "top_camera" and diameter_mm is not None:
                info_lines.append(f"Detected diameter: {diameter_px:.1f} px ≈ {diameter_mm:.1f} mm")
            if wheel_info['error'] and wheel_info['error'] != "Depth frame not available":
                 # Only show error if it's not the expected missing depth
                info_lines.append(f"Note: {wheel_info['error']}")
            
            if info_lines:
                # self.lbl_wheel_info.setText("\n".join(info_lines))
                logger.info(f"Wheel center info: {' | '.join(info_lines)}")
            else:
                # self.lbl_wheel_info.setText("Could not detect wheel center")
                logger.info("Could not detect wheel center")
            
            # Calculate statistics
            mask_pixels = np.sum(filtered_mask > 0)
            total_pixels = filtered_mask.size
            mask_percent = (mask_pixels / total_pixels) * 100
            
            status_msg = (
                f"Processing complete | "
                f"Mask pixels: {mask_pixels:,} ({mask_percent:.2f}%) | "
                f"Valid contours: {len(valid_contours)} | "
                f"Params: dev[{self.min_deviation}-{self.max_deviation}], area≥{self.min_contour_area}"
            )
            self.lbl_status.setText(status_msg)
            
            logger.info(f"Background subtraction complete: {len(valid_contours)} objects detected")
            
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to process: {e}")
            logger.error(f"Processing error: {e}", exc_info=True)


def main():
    app = QApplication(sys.argv)
    window = BackgroundSubtractionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
