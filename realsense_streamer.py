import pyrealsense2 as rs
import cv2
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RealSenseStreamer:
    """Handle RealSense camera streaming"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = None
        self.align = None
        self.is_running = False
        self.stop_flag = False
        
        self.color_frame = None
        self.depth_frame = None
    
    def initialize(self) -> bool:
        """Initialize RealSense pipeline"""
        try:
            ctx = rs.context()
            if ctx.query_devices().size() == 0:
                logger.error("No RealSense devices detected")
                return False
            
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            # Configure depth sensor
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
            
            logger.info("RealSense camera initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RealSense: {e}")
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
                    frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                    aligned_frames = self.align.process(frames)
                    
                    self.color_frame = aligned_frames.get_color_frame()
                    self.depth_frame = aligned_frames.get_depth_frame()
                    
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        continue
                    logger.error(f"Streaming error: {e}")
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
        
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
            logger.error(f"Cleanup error: {e}")


class HttpStreamer:
    """Handle HTTP/IP Camera streaming (RGB only) using OpenCV"""
    
    def __init__(self, url: str = "http://192.168.1.100:8080/video", timeout: int = 5, pixel_to_mm: float = 1.0):
        # Default URL placeholder - user might need to change this or we can add input in UI later
        self.url = url
        self.timeout = timeout
        self.pixel_to_mm = pixel_to_mm
        self.cap = None
        self.is_running = False
        self.stop_flag = False
        
        self.color_frame = None # Will be a numpy array
        self.depth_frame = None # Always None for this mode
    
    def initialize(self) -> bool:
        """Initialize VideoCapture"""
        try:
            # Using cv2.VideoCapture for HTTP stream
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open stream at {self.url}")
                return False
            
            logger.info(f"HTTP stream initialized: {self.url}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing HTTP stream: {e}")
            return False
    
    def run(self):
        """Streaming thread loop"""
        if not self.initialize():
            return
        
        self.is_running = True
        self.stop_flag = False
        
        try:
            while not self.stop_flag:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.color_frame = frame
                    else:
                        logger.warning("Failed to read frame from stream")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            self.is_running = False
            logger.info("HTTP stream stopped")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def compute_diameter_mm(self, mask: np.ndarray):
        """Compute diameter (pixels and mm) of largest contour in a binary mask.
        Returns tuple (diameter_px, diameter_mm) or (None, None) if not found."""
        if mask is None or mask.size == 0:
            return None, None

        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None

            main_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(main_contour)
            diameter_px = 2.0 * radius
            diameter_mm = diameter_px * float(self.pixel_to_mm)
            return diameter_px, diameter_mm
        except Exception as e:
            logger.error(f"Diameter computation error: {e}")
            return None, None