#!/usr/bin/env python3
"""
Real-Time Hand Gesture Control System for Racing Games Using Computer Vision
===========================================================================

An ultra-low-latency computer vision system for controlling racing games
through precise hand gestures using advanced AI and computer vision techniques.

Author: Muhammad Kashan Tariq
LinkedIn: https://www.linkedin.com/in/muhammad-kashan-tariq
Version: 1.0.0
Date: 2025-06-28
Hardware Recommended: Intel i7 or i9 cpu + NVIDIA RTX GPU + atleast 32GB DDR4 or DDR 5 ram
Software: Python 3.10, CUDA 12.1, cuDNN 8.9.7, TensorRT 8.6.1.6

TESTED WITH FORZA HORIZON 5 - COMPATIBLE WITH ALL RACING GAMES

Gesture Mappings (Exact Specifications):
========================================
1. E-BRAKE: Both hands open (fingers spread), aligned horizontally → Spacebar
2. ACCELERATE: Both hands closed (fists), aligned horizontally → W key
3. REVERSE: Right hand open, left hand closed (fist), aligned horizontally → S key
4. START/IDLE: Right hand closed (fist), left hand open, aligned horizontally → E key
5. STEERING: Hand height differences indicate steering direction
   - Left higher than right = Steer Left (A key)
   - Right higher than left = Steer Right (D key)
6. COMBINED CONTROLS: Steering can be combined with any base gesture

Performance Targets:
===================
- Latency: <50ms gesture detection to game input
- FPS: 30+ frames per second processing
- Accuracy: >95% gesture classification
- Hardware Utilization: Full GPU acceleration with CUDA
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import json
import logging
import os
import sys
import warnings
import math
import platform
import psutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto
import concurrent.futures
from collections import deque, defaultdict
import configparser
import traceback
from contextlib import contextmanager
import signal

# Suppress MediaPipe and TensorFlow warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # Enable GPU acceleration

# Game input simulation libraries with comprehensive error handling
try:
    import pynput
    from pynput import keyboard, mouse
    from pynput.keyboard import Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    print("Installing pynput for game input simulation...")
    os.system("pip install pynput")
    try:
        import pynput
        from pynput import keyboard, mouse
        from pynput.keyboard import Key, KeyCode
        PYNPUT_AVAILABLE = True
    except ImportError:
        PYNPUT_AVAILABLE = False
        print("WARNING: pynput not available - input simulation disabled")

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable failsafe for gaming
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    print("Installing pyautogui for additional input methods...")
    os.system("pip install pyautogui")
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        PYAUTOGUI_AVAILABLE = True
    except ImportError:
        PYAUTOGUI_AVAILABLE = False
        print("WARNING: pyautogui not available")

# GPU acceleration support with comprehensive CUDA detection
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_DEVICE_COUNT = torch.cuda.device_count()
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
        CUDA_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 CUDA GPU Detected: {CUDA_DEVICE_NAME}")
        print(f"🚀 CUDA Memory: {CUDA_MEMORY_GB:.1f} GB")
        print(f"🚀 PyTorch Version: {torch.__version__}")
    else:
        print("⚠️  CUDA not available - using CPU processing")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️  PyTorch not available - GPU optimizations disabled")

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Installing psutil for performance monitoring...")
    os.system("pip install psutil")
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False

class GestureType(Enum):
    """
    Comprehensive enumeration of all supported hand gestures for racing game control.
    Each gesture maps to specific in-game actions as per project specifications.
    """
    # Primary Gestures (Base Actions)
    E_BRAKE = "e_brake"                    # Both hands open → Emergency brake (Spacebar)
    ACCELERATE = "accelerate"              # Both hands closed → Move forward (W key)
    REVERSE = "reverse"                    # Right open, left closed → Move backward (S key)
    START_IDLE = "start_idle"              # Right closed, left open → Engine start/idle (E key)
    
    # Steering Gestures (Direction Modifiers)
    STEER_LEFT = "steer_left"              # Left hand higher → Turn left (A key)
    STEER_RIGHT = "steer_right"            # Right hand higher → Turn right (D key)
    
    # Combined Control Gestures (Revolutionary Feature)
    ACCELERATE_LEFT = "accelerate_left"    # Accelerate + Steer Left (W + A)
    ACCELERATE_RIGHT = "accelerate_right"  # Accelerate + Steer Right (W + D)
    REVERSE_LEFT = "reverse_left"          # Reverse + Steer Left (S + A)
    REVERSE_RIGHT = "reverse_right"        # Reverse + Steer Right (S + D)
    BRAKE_LEFT = "brake_left"              # E-brake + Steer Left (Space + A)
    BRAKE_RIGHT = "brake_right"            # E-brake + Steer Right (Space + D)
    
    # Special States
    NO_GESTURE = "no_gesture"              # No hands detected
    UNKNOWN = "unknown"                    # Hands detected but gesture unclear
    INVALID = "invalid"                    # Invalid hand configuration

class GameAction(Enum):
    """Enumeration of all possible in-game actions that can be triggered."""
    # Primary Actions
    BRAKE = "brake"                        # Emergency brake (Spacebar)
    ACCELERATE = "accelerate"              # Move forward (W key)
    REVERSE = "reverse"                    # Move backward (S key)
    STEER_LEFT = "steer_left"              # Turn left (A key)
    STEER_RIGHT = "steer_right"            # Turn right (D key)
    ENGINE_START = "engine_start"          # Start engine (E key)
    
    # Combined Actions
    ACCELERATE_AND_LEFT = "accelerate_and_left"    # W + A simultaneously
    ACCELERATE_AND_RIGHT = "accelerate_and_right"  # W + D simultaneously
    REVERSE_AND_LEFT = "reverse_and_left"          # S + A simultaneously
    REVERSE_AND_RIGHT = "reverse_and_right"        # S + D simultaneously
    BRAKE_AND_LEFT = "brake_and_left"              # Space + A simultaneously
    BRAKE_AND_RIGHT = "brake_and_right"            # Space + D simultaneously
    
    # System Actions
    NO_ACTION = "no_action"                # No input to game
    EMERGENCY_STOP = "emergency_stop"      # Emergency system stop

@dataclass
class HandLandmarkData:
    """
    Comprehensive container for hand landmark data with advanced utility methods
    for sophisticated gesture analysis and classification.
    """
    landmarks: List[Tuple[float, float, float]]  # (x, y, z) normalized coordinates
    handedness: str                               # "Left" or "Right"
    confidence: float                            # Detection confidence [0.0, 1.0]
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate landmark data on initialization."""
        if len(self.landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(self.landmarks)}")
        if self.handedness not in ["Left", "Right"]:
            raise ValueError(f"Invalid handedness: {self.handedness}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
    
    def get_finger_states(self) -> Dict[str, bool]:
        """
        Advanced finger state analysis using geometric relationships between landmarks.
        
        Returns:
            Dict mapping finger names to boolean states (True = extended, False = closed)
        """
        # MediaPipe hand landmark indices for sophisticated finger analysis
        finger_tips = [4, 8, 12, 16, 20]      # Thumb, Index, Middle, Ring, Pinky tips
        finger_joints = [3, 6, 10, 14, 18]    # Corresponding joint positions
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        finger_states = {}
        
        for i, (tip_idx, joint_idx, name) in enumerate(zip(finger_tips, finger_joints, finger_names)):
            tip = self.landmarks[tip_idx]
            joint = self.landmarks[joint_idx]
            
            if name == "thumb":
                # Thumb analysis: Use X-axis movement relative to palm center
                palm_center = self.get_palm_center()
                tip_distance = abs(tip[0] - palm_center[0])
                joint_distance = abs(joint[0] - palm_center[0])
                is_extended = tip_distance > joint_distance + 0.02
            else:
                # Other fingers: Use Y-axis movement (tip above joint for extension)
                is_extended = tip[1] < joint[1] - 0.03  # Increased threshold for reliability
            
            finger_states[name] = is_extended
        
        return finger_states
    
    def is_fist(self, threshold: float = 0.8) -> bool:
        """
        Determine if hand forms a closed fist using configurable threshold.
        
        Args:
            threshold: Confidence threshold for fist detection [0.0, 1.0]
            
        Returns:
            True if hand is classified as a fist
        """
        finger_states = self.get_finger_states()
        closed_fingers = sum(1 for extended in finger_states.values() if not extended)
        fist_confidence = closed_fingers / len(finger_states)
        return fist_confidence >= threshold
    
    def is_open_hand(self, threshold: float = 0.6) -> bool:
        """
        Determine if hand is open with configurable threshold.
        
        Args:
            threshold: Confidence threshold for open hand detection [0.0, 1.0]
            
        Returns:
            True if hand is classified as open
        """
        finger_states = self.get_finger_states()
        extended_fingers = sum(1 for extended in finger_states.values() if extended)
        open_confidence = extended_fingers / len(finger_states)
        return open_confidence >= threshold
    
    def get_palm_center(self) -> Tuple[float, float]:
        """
        Calculate precise palm center using multiple landmark points for accuracy.
        
        Returns:
            (x, y) coordinates of palm center
        """
        # Use multiple palm landmarks for more accurate center calculation
        wrist = self.landmarks[0]
        index_mcp = self.landmarks[5]
        middle_mcp = self.landmarks[9]
        ring_mcp = self.landmarks[13]
        pinky_mcp = self.landmarks[17]
        
        # Average multiple points for robust palm center
        palm_points = [wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp]
        center_x = sum(point[0] for point in palm_points) / len(palm_points)
        center_y = sum(point[1] for point in palm_points) / len(palm_points)
        
        return (center_x, center_y)
    
    def get_hand_orientation(self) -> float:
        """
        Calculate hand orientation angle for advanced gesture analysis.
        
        Returns:
            Orientation angle in degrees [-180, 180]
        """
        wrist = self.landmarks[0]
        middle_finger_mcp = self.landmarks[9]
        
        dx = middle_finger_mcp[0] - wrist[0]
        dy = middle_finger_mcp[1] - wrist[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def calculate_hand_size(self) -> float:
        """
        Estimate hand size for distance-independent gesture recognition.
        
        Returns:
            Normalized hand size metric
        """
        wrist = self.landmarks[0]
        middle_tip = self.landmarks[12]
        
        # Distance from wrist to middle finger tip as size metric
        dx = middle_tip[0] - wrist[0]
        dy = middle_tip[1] - wrist[1]
        hand_size = math.sqrt(dx*dx + dy*dy)
        
        return hand_size

@dataclass
class GestureFrame:
    """
    Comprehensive container for a complete frame of gesture analysis data
    with performance metrics and temporal information.
    """
    timestamp: float
    left_hand: Optional[HandLandmarkData]
    right_hand: Optional[HandLandmarkData]
    gesture_type: GestureType
    confidence: float
    frame_number: int
    processing_time_ms: float
    camera_fps: float = 0.0
    system_latency_ms: float = 0.0
    
    def has_both_hands(self) -> bool:
        """Check if both hands are detected in this frame."""
        return self.left_hand is not None and self.right_hand is not None
    
    def has_any_hand(self) -> bool:
        """Check if at least one hand is detected in this frame."""
        return self.left_hand is not None or self.right_hand is not None
    
    def get_hand_count(self) -> int:
        """Get the number of detected hands in this frame."""
        return sum([self.left_hand is not None, self.right_hand is not None])

class PerformanceMonitor:
    """
    Military-grade performance monitoring system for real-time tracking of system metrics,
    processing times, hardware utilization, and comprehensive performance analysis.
    """
    
    def __init__(self, window_size: int = 100, monitoring_interval: float = 1.0):
        """
        Initialize performance monitor with comprehensive metric tracking.
        
        Args:
            window_size: Number of recent measurements for moving averages
            monitoring_interval: Hardware monitoring update interval in seconds
        """
        self.window_size = window_size
        self.monitoring_interval = monitoring_interval
        
        # Performance metrics with time-based sliding windows
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.gesture_times = deque(maxlen=window_size)
        self.input_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.classification_times = deque(maxlen=window_size)
        
        # System metrics
        self.total_frames = 0
        self.total_gestures = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.peak_fps = 0.0
        self.min_fps = float('inf')
        
        # Hardware monitoring (updated in background thread)
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_used_gb = 0.0
        self.gpu_utilization = 0.0
        self.gpu_memory_percent = 0.0
        self.gpu_temperature = 0.0
        
        # Performance statistics
        self.gesture_accuracy_history = deque(maxlen=1000)
        self.latency_violations = 0  # Count of frames exceeding 50ms latency
        self.error_count = 0
        self.warning_count = 0
        
        # Threading for non-blocking hardware monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_hardware_loop, daemon=True)
        self.monitor_thread.start()
        
        # Performance thresholds for alerts
        self.fps_threshold_low = 25.0
        self.latency_threshold_ms = 50.0
        self.cpu_threshold_high = 80.0
        self.memory_threshold_high = 90.0
    
    def record_frame_time(self, frame_time: float):
        """Record frame processing time and update FPS calculations."""
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        # Update FPS calculation
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            if len(self.frame_times) > 0:
                avg_frame_time = np.mean(list(self.frame_times)[-30:])  # Use last 30 frames
                self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                
                # Update peak and minimum FPS
                if self.current_fps > self.peak_fps:
                    self.peak_fps = self.current_fps
                if self.current_fps < self.min_fps and self.current_fps > 0:
                    self.min_fps = self.current_fps
            
            self.last_fps_update = current_time
    
    def record_processing_time(self, process_time: float):
        """Record gesture processing time."""
        self.processing_times.append(process_time)
    
    def record_gesture_time(self, gesture_time: float):
        """Record gesture classification time."""
        self.gesture_times.append(gesture_time)
        self.total_gestures += 1
    
    def record_input_time(self, input_time: float):
        """Record game input simulation time."""
        self.input_times.append(input_time)
    
    def record_detection_time(self, detection_time: float):
        """Record hand detection time."""
        self.detection_times.append(detection_time)
    
    def record_classification_time(self, classification_time: float):
        """Record gesture classification time."""
        self.classification_times.append(classification_time)
    
    def record_gesture_accuracy(self, accuracy: float):
        """Record gesture recognition accuracy."""
        self.gesture_accuracy_history.append(accuracy)
    
    def record_latency_violation(self):
        """Record when system latency exceeds threshold."""
        self.latency_violations += 1
    
    def record_error(self):
        """Record system error occurrence."""
        self.error_count += 1
    
    def record_warning(self):
        """Record system warning occurrence."""
        self.warning_count += 1
    
    def _monitor_hardware_loop(self):
        """Background thread for continuous hardware monitoring."""
        while self.monitoring_active:
            try:
                if PSUTIL_AVAILABLE:
                    # CPU and memory monitoring
                    self.cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()
                    self.memory_percent = memory_info.percent
                    self.memory_used_gb = memory_info.used / (1024**3)
                
                # GPU monitoring (if CUDA available)
                if CUDA_AVAILABLE:
                    try:
                        # Get GPU utilization if available
                        if hasattr(torch.cuda, 'utilization'):
                            self.gpu_utilization = torch.cuda.utilization()
                        
                        # Get GPU memory usage
                        memory_stats = torch.cuda.memory_stats()
                        if memory_stats:
                            allocated = memory_stats.get('allocated_bytes.all.current', 0)
                            reserved = memory_stats.get('reserved_bytes.all.current', 1)
                            self.gpu_memory_percent = (allocated / reserved) * 100 if reserved > 0 else 0
                        
                        # Get GPU temperature if available via nvidia-ml-py
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            self.gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        except:
                            self.gpu_temperature = 0.0
                    except Exception:
                        self.gpu_utilization = 0.0
                        self.gpu_memory_percent = 0.0
                        self.gpu_temperature = 0.0
                
                time.sleep(self.monitoring_interval)
            except Exception:
                time.sleep(self.monitoring_interval)
                continue
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance statistics with detailed metrics.
        
        Returns:
            Dictionary containing all performance metrics and system information
        """
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate averages with error handling
        def safe_mean(data_deque):
            return np.mean(list(data_deque)) if len(data_deque) > 0 else 0.0
        
        # Calculate total system latency
        total_latency_ms = (safe_mean(self.processing_times) + 
                           safe_mean(self.gesture_times) + 
                           safe_mean(self.input_times)) * 1000
        
        stats = {
            # Performance Metrics
            'fps': {
                'current': self.current_fps,
                'peak': self.peak_fps,
                'minimum': self.min_fps if self.min_fps != float('inf') else 0.0,
                'average': safe_mean(self.frame_times) and (1.0 / safe_mean(self.frame_times))
            },
            
            # Timing Metrics (all in milliseconds)
            'timing': {
                'frame_time_ms': safe_mean(self.frame_times) * 1000,
                'processing_time_ms': safe_mean(self.processing_times) * 1000,
                'detection_time_ms': safe_mean(self.detection_times) * 1000,
                'classification_time_ms': safe_mean(self.classification_times) * 1000,
                'input_time_ms': safe_mean(self.input_times) * 1000,
                'total_latency_ms': total_latency_ms
            },
            
            # System Counters
            'counters': {
                'total_frames': self.total_frames,
                'total_gestures': self.total_gestures,
                'latency_violations': self.latency_violations,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'uptime_seconds': uptime,
                'frames_per_minute': (self.total_frames / (uptime / 60)) if uptime > 0 else 0,
                'gestures_per_minute': (self.total_gestures / (uptime / 60)) if uptime > 0 else 0
            },
            
            # Hardware Utilization
            'hardware': {
                'cpu_percent': self.cpu_percent,
                'memory_percent': self.memory_percent,
                'memory_used_gb': self.memory_used_gb,
                'gpu_utilization': self.gpu_utilization,
                'gpu_memory_percent': self.gpu_memory_percent,
                'gpu_temperature': self.gpu_temperature
            },
            
            # Performance Quality Indicators
            'quality': {
                'average_accuracy': safe_mean(self.gesture_accuracy_history),
                'latency_violation_rate': (self.latency_violations / max(self.total_frames, 1)) * 100,
                'error_rate': (self.error_count / max(self.total_frames, 1)) * 100,
                'system_health': self._calculate_system_health()
            }
        }
        
        return stats
    
    def _calculate_system_health(self) -> str:
        """
        Calculate overall system health based on multiple performance indicators.
        
        Returns:
            System health status: "EXCELLENT", "GOOD", "FAIR", "POOR", or "CRITICAL"
        """
        health_score = 100.0
        
        # FPS performance impact
        if self.current_fps < 20:
            health_score -= 30
        elif self.current_fps < 25:
            health_score -= 15
        elif self.current_fps < 30:
            health_score -= 5
        
        # Latency impact
        total_latency = (np.mean(list(self.processing_times)) + 
                        np.mean(list(self.gesture_times)) + 
                        np.mean(list(self.input_times))) * 1000 if len(self.processing_times) > 0 else 0
        
        if total_latency > 100:
            health_score -= 40
        elif total_latency > 75:
            health_score -= 25
        elif total_latency > 50:
            health_score -= 15
        
        # Hardware utilization impact
        if self.cpu_percent > 90:
            health_score -= 20
        elif self.cpu_percent > 80:
            health_score -= 10
        
        if self.memory_percent > 95:
            health_score -= 20
        elif self.memory_percent > 85:
            health_score -= 10
        
        # Error rate impact
        error_rate = (self.error_count / max(self.total_frames, 1)) * 100
        if error_rate > 5:
            health_score -= 30
        elif error_rate > 2:
            health_score -= 15
        elif error_rate > 1:
            health_score -= 5
        
        # Determine health status
        if health_score >= 90:
            return "EXCELLENT"
        elif health_score >= 75:
            return "GOOD"
        elif health_score >= 60:
            return "FAIR"
        elif health_score >= 40:
            return "POOR"
        else:
            return "CRITICAL"
    
    def get_performance_alerts(self) -> List[str]:
        """
        Generate performance alerts based on current system metrics.
        
        Returns:
            List of performance alert messages
        """
        alerts = []
        
        if self.current_fps < self.fps_threshold_low:
            alerts.append(f"LOW FPS: {self.current_fps:.1f} (target: >{self.fps_threshold_low})")
        
        total_latency = (np.mean(list(self.processing_times)) + 
                        np.mean(list(self.gesture_times)) + 
                        np.mean(list(self.input_times))) * 1000 if len(self.processing_times) > 0 else 0
        
        if total_latency > self.latency_threshold_ms:
            alerts.append(f"HIGH LATENCY: {total_latency:.1f}ms (target: <{self.latency_threshold_ms}ms)")
        
        if self.cpu_percent > self.cpu_threshold_high:
            alerts.append(f"HIGH CPU: {self.cpu_percent:.1f}% (threshold: {self.cpu_threshold_high}%)")
        
        if self.memory_percent > self.memory_threshold_high:
            alerts.append(f"HIGH MEMORY: {self.memory_percent:.1f}% (threshold: {self.memory_threshold_high}%)")
        
        return alerts
    
    def cleanup(self):
        """Stop background monitoring threads and clean up resources."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

class ConfigurationManager:
    """
    Military-grade configuration management system for comprehensive gesture recognition settings,
    game input mappings, performance parameters, and advanced system optimization options.
    """
    
    def __init__(self, config_file: str = "gesture_config.json"):
        """
        Initialize configuration manager with comprehensive default settings.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.config = self._create_comprehensive_default_config()
        self.load_configuration()
        
        # Configuration validation and auto-correction
        self._validate_and_correct_config()
    
    def _create_comprehensive_default_config(self) -> Dict[str, Any]:
        """Create comprehensive default configuration with all system parameters."""
        return {
            # Camera and Video Processing Configuration
            "camera": {
                "device_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "buffer_size": 1,
                "auto_exposure": True,
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.5,
                "backend": "auto",  # "auto", "dshow" (Windows), "v4l2" (Linux)
                "fourcc": "MJPG",   # Video codec for camera
                "auto_focus": True
            },
            
            # MediaPipe Hand Detection Advanced Configuration
            "mediapipe": {
                "model_complexity": 1,              # 0=lite, 1=full, 2=heavy
                "min_detection_confidence": 0.7,   # Hand detection threshold
                "min_tracking_confidence": 0.5,    # Hand tracking threshold
                "max_num_hands": 2,                 # Maximum hands to detect
                "static_image_mode": False,         # For video vs static images
                "refine_landmarks": True            # Enhance landmark accuracy
            },
            
            # Advanced Gesture Recognition Parameters
            "gestures": {
                "confidence_threshold": 0.85,           # Overall gesture confidence threshold
                "stability_frames": 3,                  # Frames for gesture stability
                "hand_alignment_tolerance": 0.12,       # Horizontal alignment tolerance
                "vertical_separation_threshold": 0.08,  # Steering detection threshold
                "fist_threshold": 0.8,                  # Fist detection confidence
                "open_hand_threshold": 0.6,             # Open hand detection confidence
                "debounce_time_ms": 100,                # Gesture change debounce
                "gesture_timeout_ms": 5000,             # Gesture loss timeout
                "palm_center_smoothing": 0.7,           # Palm center position smoothing
                "landmark_smoothing": 0.8,              # Landmark position smoothing
                "angle_smoothing": 0.6                  # Hand angle smoothing
            },
            
            # Comprehensive Game Control Mappings
            "game_controls": {
                # Primary Actions
                "accelerate": "w",
                "reverse": "s",
                "steer_left": "a",
                "steer_right": "d",
                "brake": "space",
                "engine_start": "e",
                "handbrake": "space",
                
                # Advanced Controls
                "boost": "x",
                "camera_change": "c",
                "horn": "h",
                "lights": "l",
                "pause": "escape",
                
                # Modifier Keys
                "shift_modifier": False,
                "ctrl_modifier": False,
                "alt_modifier": False
            },
            
            # Performance Optimization Configuration
            "performance": {
                "enable_gpu_acceleration": True,
                "max_fps": 60,
                "target_latency_ms": 25,            # Target system latency
                "threading_enabled": True,
                "thread_count": -1,                 # -1 for auto-detect
                "frame_skip_threshold": 2,
                "memory_optimization": True,
                "cpu_cores": -1,                    # -1 for auto-detect
                "priority_class": "high",           # Process priority
                "garbage_collection_interval": 100, # GC every N frames
                "enable_profiling": False           # Performance profiling
            },
            
            # Comprehensive Logging Configuration
            "logging": {
                "level": "INFO",                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
                "file_path": "gesture_control.log",
                "max_file_size_mb": 50,
                "backup_count": 5,
                "console_output": True,
                "performance_logging": True,
                "gesture_logging": True,
                "error_logging": True,
                "debug_logging": False,
                "log_format": "%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s"
            },
            
            # Display and User Interface Configuration
            "display": {
                "show_camera_feed": True,
                "show_landmarks": True,
                "show_gesture_info": True,
                "show_performance_stats": True,
                "show_debug_info": False,
                "window_size": [800, 600],
                "window_position": [100, 100],
                "fullscreen": False,
                "overlay_opacity": 0.8,
                "text_size": 0.7,
                "text_color": [0, 255, 0],          # Green text
                "background_color": [0, 0, 0],      # Black background
                "landmark_color": [255, 0, 0],      # Red landmarks
                "connection_color": [0, 255, 255]   # Cyan connections
            },
            
            # Safety and Emergency Configuration
            "safety": {
                "enable_emergency_stop": True,
                "emergency_key": "escape",
                "auto_brake_on_no_hands": True,
                "gesture_loss_timeout_ms": 1000,
                "safe_mode_enabled": False,
                "max_continuous_action_time": 30,   # Seconds
                "require_both_hands": True,
                "hand_distance_max": 0.8,           # Maximum hand separation
                "hand_distance_min": 0.1            # Minimum hand separation
            },
            
            # Advanced AI and ML Configuration
            "ai": {
                "model_optimization": True,
                "batch_processing": False,
                "use_tensorrt": True,               # Enable TensorRT optimization
                "use_cuda": True,                   # Enable CUDA acceleration
                "precision": "fp16",                # Model precision: fp32, fp16, int8
                "cache_models": True,               # Cache loaded models
                "warmup_iterations": 10             # Model warmup iterations
            },
            
            # System Monitoring Configuration
            "monitoring": {
                "enable_performance_monitoring": True,
                "monitoring_interval": 1.0,        # Hardware monitoring interval
                "alert_thresholds": {
                    "fps_min": 25.0,
                    "latency_max_ms": 50.0,
                    "cpu_max_percent": 85.0,
                    "memory_max_percent": 90.0,
                    "gpu_temp_max_celsius": 85.0
                },
                "save_performance_logs": True,
                "performance_log_interval": 60     # Save performance logs every N seconds
            }
        }
    
    def load_configuration(self):
        """Load configuration from file with comprehensive error handling."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._deep_merge_configs(self.config, file_config)
                print(f"✅ Configuration loaded from {self.config_file}")
            else:
                self.save_configuration()
                print(f"✅ Default configuration created: {self.config_file}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            print("🔧 Using default configuration")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("🔧 Using default configuration")
    
    def save_configuration(self):
        """Save current configuration to file with proper formatting."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, sort_keys=True, ensure_ascii=False)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def _deep_merge_configs(self, default: Dict, user: Dict):
        """Recursively merge user configuration with defaults."""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._deep_merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                # Add new keys from user config
                default[key] = value
    
    def _validate_and_correct_config(self):
        """Validate configuration values and auto-correct invalid settings."""
        corrections = []
        
        # Validate camera settings
        if not isinstance(self.config["camera"]["device_id"], int) or self.config["camera"]["device_id"] < 0:
            self.config["camera"]["device_id"] = 0
            corrections.append("camera.device_id corrected to 0")
        
        # Validate MediaPipe settings
        if not 0 <= self.config["mediapipe"]["model_complexity"] <= 2:
            self.config["mediapipe"]["model_complexity"] = 1
            corrections.append("mediapipe.model_complexity corrected to 1")
        
        # Validate gesture thresholds
        for threshold_key in ["confidence_threshold", "fist_threshold", "open_hand_threshold"]:
            if not 0.0 <= self.config["gestures"][threshold_key] <= 1.0:
                self.config["gestures"][threshold_key] = 0.8
                corrections.append(f"gestures.{threshold_key} corrected to 0.8")
        
        # Validate performance settings
        if self.config["performance"]["max_fps"] <= 0:
            self.config["performance"]["max_fps"] = 30
            corrections.append("performance.max_fps corrected to 30")
        
        if corrections:
            print(f"🔧 Configuration corrections applied: {len(corrections)} items")
            for correction in corrections:
                print(f"   • {correction}")
            self.save_configuration()
    
    def get(self, path: str, default=None):
        """
        Get configuration value using dot notation path with comprehensive error handling.
        
        Args:
            path: Dot-separated path (e.g., "camera.width")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = path.split('.')
            value = self.config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError, AttributeError):
            return default
    
    def set(self, path: str, value: Any, save: bool = False):
        """
        Set configuration value using dot notation path.
        
        Args:
            path: Dot-separated path (e.g., "camera.width")
            value: Value to set
            save: Whether to save configuration to file immediately
        """
        try:
            keys = path.split('.')
            config_ref = self.config
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            config_ref[keys[-1]] = value
            
            if save:
                self.save_configuration()
        except Exception as e:
            print(f"❌ Error setting config path {path}: {e}")
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera-specific configuration."""
        return self.config.get("camera", {})
    
    def get_mediapipe_config(self) -> Dict[str, Any]:
        """Get MediaPipe-specific configuration."""
        return self.config.get("mediapipe", {})
    
    def get_gesture_config(self) -> Dict[str, Any]:
        """Get gesture recognition configuration."""
        return self.config.get("gestures", {})
    
    def get_game_controls_config(self) -> Dict[str, Any]:
        """Get game controls configuration."""
        return self.config.get("game_controls", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration."""
        return self.config.get("performance", {})

class GameInputController:
    """
    Military-grade game input simulation system with ultra-low latency key press/release handling,
    simultaneous input support, advanced error recovery, and comprehensive input state management.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize game input controller with comprehensive configuration and safety systems.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.is_available = PYNPUT_AVAILABLE
        
        if not self.is_available:
            print("❌ Input controller not available - pynput library missing")
            return
        
        # Initialize input controllers
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller() if hasattr(mouse, 'Controller') else None
        
        # Input state management
        self.active_keys = set()              # Currently pressed keys
        self.last_action_time = {}            # Timing for each action
        self.action_start_time = {}           # When each action started
        self.input_lock = threading.Lock()    # Thread safety
        
        # Configuration parameters
        self.debounce_time = config.get("gestures.debounce_time_ms", 100) / 1000.0
        self.max_action_time = config.get("safety.max_continuous_action_time", 30)
        
        # Key mapping from configuration with comprehensive fallbacks
        self.key_mapping = self._initialize_key_mapping(config)
        self.processed_keys = self._process_key_mapping()
        
        # Safety and monitoring
        self.emergency_stop_active = False
        self.total_inputs_sent = 0
        self.failed_inputs = 0
        self.input_history = deque(maxlen=1000)
        
        # Performance monitoring
        self.input_latency_history = deque(maxlen=100)
        self.last_successful_input = time.time()
        
        print("🎮 Game Input Controller initialized")
        if self.is_available:
            self._print_key_mappings()
        
    def _initialize_key_mapping(self, config: ConfigurationManager) -> Dict[GameAction, str]:
        """Initialize comprehensive key mapping from configuration."""
        return {
            # Primary actions
            GameAction.ACCELERATE: config.get("game_controls.accelerate", "w"),
            GameAction.REVERSE: config.get("game_controls.reverse", "s"),
            GameAction.STEER_LEFT: config.get("game_controls.steer_left", "a"),
            GameAction.STEER_RIGHT: config.get("game_controls.steer_right", "d"),
            GameAction.BRAKE: config.get("game_controls.brake", "space"),
            GameAction.ENGINE_START: config.get("game_controls.engine_start", "e"),
            
            # Combined actions use the same keys as primary actions
            GameAction.ACCELERATE_AND_LEFT: [config.get("game_controls.accelerate", "w"), 
                                            config.get("game_controls.steer_left", "a")],
            GameAction.ACCELERATE_AND_RIGHT: [config.get("game_controls.accelerate", "w"), 
                                             config.get("game_controls.steer_right", "d")],
            GameAction.REVERSE_AND_LEFT: [config.get("game_controls.reverse", "s"), 
                                         config.get("game_controls.steer_left", "a")],
            GameAction.REVERSE_AND_RIGHT: [config.get("game_controls.reverse", "s"), 
                                          config.get("game_controls.steer_right", "d")],
            GameAction.BRAKE_AND_LEFT: [config.get("game_controls.brake", "space"), 
                                       config.get("game_controls.steer_left", "a")],
            GameAction.BRAKE_AND_RIGHT: [config.get("game_controls.brake", "space"), 
                                        config.get("game_controls.steer_right", "d")]
        }
    
    def _process_key_mapping(self) -> Dict[GameAction, Union[Any, List[Any]]]:
        """Convert string keys to pynput Key objects with comprehensive error handling."""
        processed_keys = {}
        
        for action, key_config in self.key_mapping.items():
            try:
                if isinstance(key_config, list):
                    # Handle combined actions (multiple keys)
                    processed_keys[action] = [self._string_to_key(key_str) for key_str in key_config]
                else:
                    # Handle single key actions
                    processed_keys[action] = self._string_to_key(key_config)
            except Exception as e:
                print(f"⚠️  Warning: Failed to process key mapping for {action.value}: {e}")
                # Use default fallback
                processed_keys[action] = KeyCode.from_char('x')
        
        return processed_keys
    
    def _string_to_key(self, key_str: str) -> Union[Key, KeyCode]:
        """Convert string representation to pynput Key object."""
        key_str = key_str.lower().strip()
        
        # Special keys
        special_keys = {
            "space": Key.space,
            "enter": Key.enter,
            "escape": Key.esc,
            "tab": Key.tab,
            "shift": Key.shift,
            "ctrl": Key.ctrl,
            "alt": Key.alt,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
            "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
            "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12
        }
        
        if key_str in special_keys:
            return special_keys[key_str]
        elif len(key_str) == 1 and key_str.isalpha():
            return KeyCode.from_char(key_str)
        elif len(key_str) == 1 and key_str.isdigit():
            return KeyCode.from_char(key_str)
        else:
            # Fallback for unknown keys
            try:
                return getattr(Key, key_str)
            except AttributeError:
                return KeyCode.from_char(key_str[0] if key_str else 'x')
    
    def _print_key_mappings(self):
        """Print current key mappings for user reference."""
        print("🔧 Game Control Key Mappings:")
        for action, key_config in self.key_mapping.items():
            if isinstance(key_config, list):
                key_str = " + ".join(str(k) for k in key_config)
            else:
                key_str = str(key_config)
            print(f"   {action.value}: {key_str}")
    
    def press_key(self, key: Union[Key, KeyCode]) -> bool:
        """
        Press a single key with comprehensive error handling and state tracking.
        
        Args:
            key: pynput Key or KeyCode object to press
            
        Returns:
            True if key was pressed successfully, False otherwise
        """
        if not self.is_available or self.emergency_stop_active:
            return False
        
        start_time = time.time()
        
        with self.input_lock:
            try:
                # Only press if not already pressed
                if key not in self.active_keys:
                    self.keyboard_controller.press(key)
                    self.active_keys.add(key)
                    self.last_successful_input = time.time()
                
                # Record input timing
                input_latency = (time.time() - start_time) * 1000
                self.input_latency_history.append(input_latency)
                
                return True
                
            except Exception as e:
                print(f"❌ Error pressing key {key}: {e}")
                self.failed_inputs += 1
                return False
    
    def release_key(self, key: Union[Key, KeyCode]) -> bool:
        """
        Release a single key with comprehensive error handling.
        
        Args:
            key: pynput Key or KeyCode object to release
            
        Returns:
            True if key was released successfully, False otherwise
        """
        if not self.is_available:
            return False
        
        with self.input_lock:
            try:
                # Only release if currently pressed
                if key in self.active_keys:
                    self.keyboard_controller.release(key)
                    self.active_keys.remove(key)
                
                return True
                
            except Exception as e:
                print(f"❌ Error releasing key {key}: {e}")
                self.failed_inputs += 1
                return False
    
    def press_multiple_keys(self, keys: List[Union[Key, KeyCode]]) -> bool:
        """
        Press multiple keys simultaneously for combined actions.
        
        Args:
            keys: List of keys to press simultaneously
            
        Returns:
            True if all keys were pressed successfully, False otherwise
        """
        if not self.is_available or self.emergency_stop_active:
            return False
        
        success_count = 0
        for key in keys:
            if self.press_key(key):
                success_count += 1
        
        return success_count == len(keys)
    
    def release_multiple_keys(self, keys: List[Union[Key, KeyCode]]) -> bool:
        """
        Release multiple keys simultaneously.
        
        Args:
            keys: List of keys to release
            
        Returns:
            True if all keys were released successfully, False otherwise
        """
        if not self.is_available:
            return False
        
        success_count = 0
        for key in keys:
            if self.release_key(key):
                success_count += 1
        
        return success_count == len(keys)
    
    def execute_gesture_action(self, gesture: GestureType) -> List[GameAction]:
        """
        Execute the appropriate game actions based on detected gesture with comprehensive mapping.
        
        Args:
            gesture: Detected gesture type
            
        Returns:
            List of game actions that were executed
        """
        if not self.is_available or self.emergency_stop_active:
            return []
        
        start_time = time.time()
        executed_actions = []
        
        # Release all keys first to ensure clean state
        self._release_all_keys()
        
        # Map gestures to game actions and execute them
        try:
            if gesture == GestureType.E_BRAKE:
                if self._execute_single_action(GameAction.BRAKE):
                    executed_actions.append(GameAction.BRAKE)
            
            elif gesture == GestureType.ACCELERATE:
                if self._execute_single_action(GameAction.ACCELERATE):
                    executed_actions.append(GameAction.ACCELERATE)
            
            elif gesture == GestureType.REVERSE:
                if self._execute_single_action(GameAction.REVERSE):
                    executed_actions.append(GameAction.REVERSE)
            
            elif gesture == GestureType.START_IDLE:
                if self._execute_single_action(GameAction.ENGINE_START):
                    executed_actions.append(GameAction.ENGINE_START)
            
            elif gesture == GestureType.ACCELERATE_LEFT:
                if self._execute_combined_action(GameAction.ACCELERATE_AND_LEFT):
                    executed_actions.append(GameAction.ACCELERATE_AND_LEFT)
            
            elif gesture == GestureType.ACCELERATE_RIGHT:
                if self._execute_combined_action(GameAction.ACCELERATE_AND_RIGHT):
                    executed_actions.append(GameAction.ACCELERATE_AND_RIGHT)
            
            elif gesture == GestureType.REVERSE_LEFT:
                if self._execute_combined_action(GameAction.REVERSE_AND_LEFT):
                    executed_actions.append(GameAction.REVERSE_AND_LEFT)
            
            elif gesture == GestureType.REVERSE_RIGHT:
                if self._execute_combined_action(GameAction.REVERSE_AND_RIGHT):
                    executed_actions.append(GameAction.REVERSE_AND_RIGHT)
            
            elif gesture == GestureType.BRAKE_LEFT:
                if self._execute_combined_action(GameAction.BRAKE_AND_LEFT):
                    executed_actions.append(GameAction.BRAKE_AND_LEFT)
            
            elif gesture == GestureType.BRAKE_RIGHT:
                if self._execute_combined_action(GameAction.BRAKE_AND_RIGHT):
                    executed_actions.append(GameAction.BRAKE_AND_RIGHT)
            
            elif gesture in [GestureType.NO_GESTURE, GestureType.UNKNOWN, GestureType.INVALID]:
                # Apply safety brake if configured
                if self.config.get("safety.auto_brake_on_no_hands", True):
                    if self._execute_single_action(GameAction.BRAKE):
                        executed_actions.append(GameAction.BRAKE)
        
        except Exception as e:
            print(f"❌ Error executing gesture action for {gesture.value}: {e}")
            # Emergency brake on error
            self._execute_single_action(GameAction.BRAKE)
            executed_actions = [GameAction.EMERGENCY_STOP]
        
        # Record execution metrics
        execution_time = time.time() - start_time
        self.total_inputs_sent += len(executed_actions)
        
        # Add to input history
        self.input_history.append({
            'timestamp': start_time,
            'gesture': gesture.value,
            'actions': [action.value for action in executed_actions],
            'execution_time_ms': execution_time * 1000
        })
        
        return executed_actions
    
    def _execute_single_action(self, action: GameAction) -> bool:
        """Execute a single game action."""
        if action not in self.processed_keys:
            return False
        
        key = self.processed_keys[action]
        return self.press_key(key)
    
    def _execute_combined_action(self, action: GameAction) -> bool:
        """Execute a combined game action (multiple keys)."""
        if action not in self.processed_keys:
            return False
        
        keys = self.processed_keys[action]
        if isinstance(keys, list):
            return self.press_multiple_keys(keys)
        else:
            return self.press_key(keys)
    
    def _release_all_keys(self):
        """Release all currently pressed keys to ensure clean state."""
        with self.input_lock:
            keys_to_release = list(self.active_keys)
            for key in keys_to_release:
                try:
                    self.keyboard_controller.release(key)
                    self.active_keys.remove(key)
                except Exception:
                    continue
    
    def emergency_stop(self):
        """Emergency stop function - releases all keys and applies brake."""
        print("🚨 EMERGENCY STOP ACTIVATED")
        self.emergency_stop_active = True
        self._release_all_keys()
        
        # Apply emergency brake
        try:
            brake_key = self.processed_keys.get(GameAction.BRAKE)
            if brake_key:
                self.keyboard_controller.press(brake_key)
                time.sleep(0.1)  # Hold brake briefly
                self.keyboard_controller.release(brake_key)
        except Exception:
            pass
    
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        self.emergency_stop_active = False
        print("✅ Emergency stop reset - system ready")
    
    def get_input_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive input controller statistics.
        
        Returns:
            Dictionary containing input statistics and performance metrics
        """
        avg_latency = np.mean(self.input_latency_history) if self.input_latency_history else 0.0
        
        return {
            'total_inputs_sent': self.total_inputs_sent,
            'failed_inputs': self.failed_inputs,
            'success_rate': (self.total_inputs_sent - self.failed_inputs) / max(self.total_inputs_sent, 1) * 100,
            'average_input_latency_ms': avg_latency,
            'active_keys_count': len(self.active_keys),
            'emergency_stop_active': self.emergency_stop_active,
            'input_history_size': len(self.input_history),
            'time_since_last_input': time.time() - self.last_successful_input
        }
    
    def cleanup(self):
        """Clean up input controller and release all keys."""
        print("🧹 Cleaning up input controller...")
        self._release_all_keys()
        self.emergency_stop_active = True

class HandGestureRecognizer:
    """
    Military-grade hand gesture recognition system using MediaPipe with advanced
    geometric analysis, temporal smoothing, and sophisticated gesture classification.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize hand gesture recognizer with MediaPipe and advanced configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        
        # Initialize MediaPipe hands solution with comprehensive error handling
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            raise
        
        # Configure MediaPipe hands with version compatibility
        self.hands = self._initialize_mediapipe_hands(config)
        
        # Gesture recognition parameters
        self.confidence_threshold = config.get("gestures.confidence_threshold", 0.85)
        self.hand_alignment_tolerance = config.get("gestures.hand_alignment_tolerance", 0.12)
        self.vertical_separation_threshold = config.get("gestures.vertical_separation_threshold", 0.08)
        self.fist_threshold = config.get("gestures.fist_threshold", 0.8)
        self.open_hand_threshold = config.get("gestures.open_hand_threshold", 0.6)
        
        # Advanced temporal smoothing and stability
        self.stability_frames = config.get("gestures.stability_frames", 3)
        self.recent_gestures = deque(maxlen=self.stability_frames * 2)
        self.gesture_confidence_history = deque(maxlen=20)
        self.palm_center_history = {'left': deque(maxlen=10), 'right': deque(maxlen=10)}
        
        # Performance tracking
        self.total_detections = 0
        self.successful_detections = 0
        self.gesture_classification_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        
        # Advanced gesture analysis
        self.hand_size_history = {'left': deque(maxlen=20), 'right': deque(maxlen=20)}
        self.hand_orientation_history = {'left': deque(maxlen=10), 'right': deque(maxlen=10)}
        
        print("🤖 Hand Gesture Recognizer initialized with MediaPipe")
        self._print_mediapipe_info()
    
    def _initialize_mediapipe_hands(self, config: ConfigurationManager):
        """Initialize MediaPipe Hands with version compatibility."""
        # Get configuration parameters
        mediapipe_config = config.get_mediapipe_config()
        
        # Build hands configuration with version compatibility
        hands_params = {
            'static_image_mode': mediapipe_config.get('static_image_mode', False),
            'max_num_hands': mediapipe_config.get('max_num_hands', 2),
            'min_detection_confidence': mediapipe_config.get('min_detection_confidence', 0.7),
            'min_tracking_confidence': mediapipe_config.get('min_tracking_confidence', 0.5)
        }
        
        # Try to add model_complexity if supported
        try:
            hands_params['model_complexity'] = mediapipe_config.get('model_complexity', 1)
            hands = self.mp_hands.Hands(**hands_params)
            self.model_complexity = mediapipe_config.get('model_complexity', 1)
        except TypeError as e:
            if 'model_complexity' in str(e):
                # Remove model_complexity for older MediaPipe versions
                hands_params.pop('model_complexity', None)
                hands = self.mp_hands.Hands(**hands_params)
                self.model_complexity = "not_supported"
                print("ℹ️  Using MediaPipe version without model_complexity parameter")
            else:
                raise e
        
        return hands
    
    def _print_mediapipe_info(self):
        """Print MediaPipe configuration information."""
        try:
            print(f"   📊 MediaPipe Version: {mp.__version__}")
            print(f"   🧠 Model Complexity: {self.model_complexity}")
            print(f"   🎯 Detection Confidence: {self.config.get('mediapipe.min_detection_confidence', 0.7)}")
            print(f"   📍 Tracking Confidence: {self.config.get('mediapipe.min_tracking_confidence', 0.5)}")
            print(f"   ✋ Max Hands: {self.config.get('mediapipe.max_num_hands', 2)}")
        except Exception as e:
            print(f"⚠️  Could not display MediaPipe info: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandLandmarkData], np.ndarray]:
        """
        Process video frame to detect hand landmarks with optimized performance.
        
        Args:
            frame: RGB image frame from camera
            
        Returns:
            Tuple of (detected hands list, annotated frame)
        """
        start_time = time.time()
        self.total_detections += 1
        
        try:
            # Convert BGR to RGB for MediaPipe
            if frame.shape[2] == 3:  # Ensure it's a color image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Performance optimization: make frame non-writeable during processing
            rgb_frame.flags.writeable = False
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV display
            rgb_frame.flags.writeable = True
            if frame.shape[2] == 3:
                output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            else:
                output_frame = rgb_frame.copy()
            
            detected_hands = []
            
            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    try:
                        # Extract landmark coordinates
                        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        
                        # Create HandLandmarkData object
                        hand_data = HandLandmarkData(
                            landmarks=landmarks,
                            handedness=handedness.classification[0].label,
                            confidence=handedness.classification[0].score
                        )
                        detected_hands.append(hand_data)
                        
                        # Draw landmarks if enabled
                        if self.config.get("display.show_landmarks", True):
                            self._draw_enhanced_landmarks(output_frame, hand_landmarks, hand_data.handedness)
                    
                    except Exception as e:
                        print(f"⚠️  Error processing hand landmarks: {e}")
                        continue
            
            self.successful_detections += 1
            
        except Exception as e:
            print(f"❌ Error in frame processing: {e}")
            detected_hands = []
            output_frame = frame.copy()
        
        # Record processing time
        processing_time = time.time() - start_time
        self.detection_times.append(processing_time)
        
        return detected_hands, output_frame
    
    def _draw_enhanced_landmarks(self, frame: np.ndarray, hand_landmarks, handedness: str):
        """Draw enhanced hand landmarks with improved visualization."""
        try:
            # Get colors from configuration
            landmark_color = tuple(self.config.get("display.landmark_color", [255, 0, 0]))
            connection_color = tuple(self.config.get("display.connection_color", [0, 255, 255]))
            
            # Draw hand connections
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )
            
            # Add handedness label
            if len(hand_landmarks.landmark) > 0:
                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                label_x = int(wrist.x * w)
                label_y = int(wrist.y * h) - 20
                
                cv2.putText(frame, f"{handedness} Hand", (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, landmark_color, 2)
        
        except Exception as e:
            print(f"⚠️  Error drawing landmarks: {e}")
    
    def classify_gesture(self, hands: List[HandLandmarkData]) -> Tuple[GestureType, float]:
        """
        Classify gesture using advanced geometric analysis and machine learning techniques.
        
        Args:
            hands: List of detected hand landmarks
            
        Returns:
            Tuple of (gesture type, confidence score)
        """
        start_time = time.time()
        
        try:
            if len(hands) == 0:
                return GestureType.NO_GESTURE, 0.0
            
            if len(hands) == 1:
                # Single hand detected - limited gesture set with safety defaults
                return self._classify_single_hand_gesture(hands[0])
            
            # Two hands detected - full gesture classification
            gesture, confidence = self._classify_two_hand_gesture(hands)
            
            # Apply temporal smoothing for stability
            stable_gesture, stable_confidence = self._apply_temporal_smoothing(gesture, confidence)
            
            # Record classification time
            classification_time = time.time() - start_time
            self.gesture_classification_times.append(classification_time)
            
            return stable_gesture, stable_confidence
        
        except Exception as e:
            print(f"❌ Error in gesture classification: {e}")
            return GestureType.UNKNOWN, 0.0
    
    def _classify_single_hand_gesture(self, hand: HandLandmarkData) -> Tuple[GestureType, float]:
        """
        Classify gestures when only one hand is detected (safety-focused).
        
        Args:
            hand: Single hand landmark data
            
        Returns:
            Tuple of (gesture type, confidence score)
        """
        # For safety, default to brake when only one hand is visible
        if hand.is_open_hand(self.open_hand_threshold):
            return GestureType.E_BRAKE, 0.6
        elif hand.is_fist(self.fist_threshold):
            return GestureType.E_BRAKE, 0.5  # Conservative safety approach
        
        return GestureType.UNKNOWN, 0.3
    
    def _classify_two_hand_gesture(self, hands: List[HandLandmarkData]) -> Tuple[GestureType, float]:
        """
        Classify gestures when both hands are detected using comprehensive analysis.
        
        Args:
            hands: List of two hand landmark data objects
            
        Returns:
            Tuple of (gesture type, confidence score)
        """
        if len(hands) != 2:
            return GestureType.UNKNOWN, 0.0
        
        # Identify left and right hands
        left_hand = None
        right_hand = None
        
        for hand in hands:
            if hand.handedness == "Left":
                left_hand = hand
            elif hand.handedness == "Right":
                right_hand = hand
        
        if not (left_hand and right_hand):
            return GestureType.UNKNOWN, 0.0
        
        # Advanced gesture analysis
        gesture_analysis = self._perform_advanced_gesture_analysis(left_hand, right_hand)
        
        return self._determine_gesture_from_analysis(gesture_analysis)
    
    def _perform_advanced_gesture_analysis(self, left_hand: HandLandmarkData, 
                                         right_hand: HandLandmarkData) -> Dict[str, Any]:
        """
        Perform comprehensive gesture analysis with multiple geometric features.
        
        Args:
            left_hand: Left hand landmark data
            right_hand: Right hand landmark data
            
        Returns:
            Dictionary containing all analysis results
        """
        analysis = {}
        
        # Basic hand state analysis
        analysis['left_is_fist'] = left_hand.is_fist(self.fist_threshold)
        analysis['left_is_open'] = left_hand.is_open_hand(self.open_hand_threshold)
        analysis['right_is_fist'] = right_hand.is_fist(self.fist_threshold)
        analysis['right_is_open'] = right_hand.is_open_hand(self.open_hand_threshold)
        
        # Palm center analysis with smoothing
        left_center = left_hand.get_palm_center()
        right_center = right_hand.get_palm_center()
        
        # Apply temporal smoothing to palm centers
        self.palm_center_history['left'].append(left_center)
        self.palm_center_history['right'].append(right_center)
        
        if len(self.palm_center_history['left']) > 1:
            smoothed_left = self._smooth_palm_center('left')
            smoothed_right = self._smooth_palm_center('right')
        else:
            smoothed_left = left_center
            smoothed_right = right_center
        
        analysis['left_center'] = smoothed_left
        analysis['right_center'] = smoothed_right
        
        # Hand alignment analysis
        vertical_diff = abs(smoothed_left[1] - smoothed_right[1])
        analysis['hands_aligned'] = vertical_diff < self.hand_alignment_tolerance
        analysis['vertical_difference'] = vertical_diff
        
        # Steering direction analysis
        analysis['steering_left'] = smoothed_left[1] < (smoothed_right[1] - self.vertical_separation_threshold)
        analysis['steering_right'] = smoothed_right[1] < (smoothed_left[1] - self.vertical_separation_threshold)
        analysis['no_steering'] = not (analysis['steering_left'] or analysis['steering_right'])
        
        # Hand distance analysis
        hand_distance = math.sqrt((smoothed_left[0] - smoothed_right[0])**2 + 
                                 (smoothed_left[1] - smoothed_right[1])**2)
        analysis['hand_distance'] = hand_distance
        analysis['hands_too_far'] = hand_distance > self.config.get("safety.hand_distance_max", 0.8)
        analysis['hands_too_close'] = hand_distance < self.config.get("safety.hand_distance_min", 0.1)
        
        # Confidence calculation based on hand detection quality
        analysis['base_confidence'] = min(left_hand.confidence, right_hand.confidence)
        
        # Hand size consistency check
        left_size = left_hand.calculate_hand_size()
        right_size = right_hand.calculate_hand_size()
        size_ratio = left_size / right_size if right_size > 0 else 1.0
        analysis['size_consistency'] = 0.5 <= size_ratio <= 2.0  # Reasonable size ratio
        
        return analysis
    
    def _smooth_palm_center(self, hand_side: str) -> Tuple[float, float]:
        """Apply temporal smoothing to palm center coordinates."""
        history = self.palm_center_history[hand_side]
        if len(history) == 0:
            return (0.5, 0.5)
        
        # Weighted average with more weight on recent positions
        weights = np.linspace(0.5, 1.0, len(history))
        weights = weights / np.sum(weights)
        
        x_coords = [center[0] for center in history]
        y_coords = [center[1] for center in history]
        
        smoothed_x = np.average(x_coords, weights=weights)
        smoothed_y = np.average(y_coords, weights=weights)
        
        return (smoothed_x, smoothed_y)
    
    def _determine_gesture_from_analysis(self, analysis: Dict[str, Any]) -> Tuple[GestureType, float]:
        """
        Determine gesture type from comprehensive analysis results.
        
        Args:
            analysis: Gesture analysis results
            
        Returns:
            Tuple of (gesture type, confidence score)
        """
        base_confidence = analysis['base_confidence']
        
        # Check for invalid hand configurations
        if analysis['hands_too_far'] or analysis['hands_too_close']:
            return GestureType.INVALID, base_confidence * 0.3
        
        # Determine base gesture (without steering)
        base_gesture, gesture_confidence = self._classify_base_gesture(analysis)
        
        # Apply steering modifications for combined gestures
        if analysis['steering_left']:
            combined_gesture = self._get_combined_gesture_left(base_gesture)
            if combined_gesture != base_gesture:
                return combined_gesture, gesture_confidence * 0.95
        elif analysis['steering_right']:
            combined_gesture = self._get_combined_gesture_right(base_gesture)
            if combined_gesture != base_gesture:
                return combined_gesture, gesture_confidence * 0.95
        
        # Return base gesture if no steering
        return base_gesture, gesture_confidence
    
    def _classify_base_gesture(self, analysis: Dict[str, Any]) -> Tuple[GestureType, float]:
        """Classify the base gesture without steering modifiers."""
        base_confidence = analysis['base_confidence']
        
        left_fist = analysis['left_is_fist']
        left_open = analysis['left_is_open']
        right_fist = analysis['right_is_fist']
        right_open = analysis['right_is_open']
        
        # Apply size consistency bonus
        confidence_multiplier = 1.1 if analysis['size_consistency'] else 0.9
        
        if left_open and right_open:
            return GestureType.E_BRAKE, base_confidence * 0.95 * confidence_multiplier
        elif left_fist and right_fist:
            return GestureType.ACCELERATE, base_confidence * 1.0 * confidence_multiplier
        elif left_fist and right_open:
            return GestureType.REVERSE, base_confidence * 0.9 * confidence_multiplier
        elif left_open and right_fist:
            return GestureType.START_IDLE, base_confidence * 0.85 * confidence_multiplier
        else:
            # Ambiguous hand states
            return GestureType.UNKNOWN, base_confidence * 0.4
    
    def _get_combined_gesture_left(self, base_gesture: GestureType) -> GestureType:
        """Get combined gesture for left steering."""
        gesture_mapping = {
            GestureType.E_BRAKE: GestureType.BRAKE_LEFT,
            GestureType.ACCELERATE: GestureType.ACCELERATE_LEFT,
            GestureType.REVERSE: GestureType.REVERSE_LEFT
        }
        return gesture_mapping.get(base_gesture, base_gesture)
    
    def _get_combined_gesture_right(self, base_gesture: GestureType) -> GestureType:
        """Get combined gesture for right steering."""
        gesture_mapping = {
            GestureType.E_BRAKE: GestureType.BRAKE_RIGHT,
            GestureType.ACCELERATE: GestureType.ACCELERATE_RIGHT,
            GestureType.REVERSE: GestureType.REVERSE_RIGHT
        }
        return gesture_mapping.get(base_gesture, base_gesture)
    
    def _apply_temporal_smoothing(self, gesture: GestureType, confidence: float) -> Tuple[GestureType, float]:
        """
        Apply temporal smoothing for gesture stability with advanced algorithms.
        
        Args:
            gesture: Current detected gesture
            confidence: Current confidence score
            
        Returns:
            Tuple of (smoothed gesture, smoothed confidence)
        """
        # Add current gesture to history
        self.recent_gestures.append((gesture, confidence))
        self.gesture_confidence_history.append(confidence)
        
        # Require minimum history for smoothing
        if len(self.recent_gestures) < self.stability_frames:
            return gesture, confidence * 0.8  # Reduced confidence for unstable gestures
        
        # Analyze gesture consistency
        recent_gesture_types = [g[0] for g in list(self.recent_gestures)]
        gesture_counts = {}
        
        for g in recent_gesture_types:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Find most frequent gesture
        most_frequent_gesture = max(gesture_counts, key=gesture_counts.get)
        frequency_ratio = gesture_counts[most_frequent_gesture] / len(recent_gesture_types)
        
        # Calculate confidence based on consistency and recent confidence
        if frequency_ratio >= 0.7:  # High consistency threshold
            # Use weighted average of recent confidences for this gesture
            relevant_confidences = [conf for g, conf in self.recent_gestures 
                                  if g == most_frequent_gesture]
            avg_confidence = np.mean(relevant_confidences)
            smoothed_confidence = avg_confidence * (0.8 + 0.2 * frequency_ratio)
            
            return most_frequent_gesture, min(smoothed_confidence, 1.0)
        else:
            # Low consistency - return current gesture with reduced confidence
            return gesture, confidence * 0.6
    
    def get_recognition_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive gesture recognition statistics.
        
        Returns:
            Dictionary containing recognition performance metrics
        """
        avg_detection_time = np.mean(self.detection_times) * 1000 if self.detection_times else 0.0
        avg_classification_time = np.mean(self.gesture_classification_times) * 1000 if self.gesture_classification_times else 0.0
        
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'detection_success_rate': (self.successful_detections / max(self.total_detections, 1)) * 100,
            'average_detection_time_ms': avg_detection_time,
            'average_classification_time_ms': avg_classification_time,
            'gesture_stability_frames': self.stability_frames,
            'confidence_threshold': self.confidence_threshold,
            'recent_gesture_count': len(self.recent_gestures)
        }
    
    def cleanup(self):
        """Clean up MediaPipe resources and stop processing."""
        try:
            if hasattr(self, 'hands') and self.hands:
                self.hands.close()
            print("🧹 Hand Gesture Recognizer cleaned up")
        except Exception as e:
            print(f"⚠️  Error during gesture recognizer cleanup: {e}")

class CameraManager:
    """
    Military-grade camera management system with advanced device detection,
    optimized settings configuration, and robust error handling for gaming performance.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize camera manager with comprehensive configuration and auto-detection.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.camera = None
        self.is_initialized = False
        
        # Camera configuration
        camera_config = config.get_camera_config()
        self.device_id = camera_config.get('device_id', 0)
        self.width = camera_config.get('width', 1280)
        self.height = camera_config.get('height', 720)
        self.fps = camera_config.get('fps', 30)
        self.buffer_size = camera_config.get('buffer_size', 1)
        self.backend = camera_config.get('backend', 'auto')
        self.fourcc = camera_config.get('fourcc', 'MJPG')
        
        # Performance optimization
        self.frame_skip_count = 0
        self.frame_skip_threshold = config.get("performance.frame_skip_threshold", 2)
        self.last_frame = None
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Camera statistics
        self.capture_times = deque(maxlen=100)
        self.frame_rate_history = deque(maxlen=50)
        self.last_fps_calculation = time.time()
        self.actual_fps = 0.0
        
        print("📹 Camera Manager initialized")
    
    def detect_available_cameras(self) -> List[int]:
        """
        Detect all available camera devices on the system.
        
        Returns:
            List of available camera device indices
        """
        available_cameras = []
        
        print("🔍 Detecting available cameras...")
        
        # Test camera indices 0-5 (covers most systems)
        for i in range(6):
            try:
                if platform.system() == "Windows":
                    test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    test_cap = cv2.VideoCapture(i)
                
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        print(f"   ✅ Camera {i}: Available")
                    else:
                        print(f"   ❌ Camera {i}: Cannot read frames")
                else:
                    print(f"   ❌ Camera {i}: Cannot open")
                
                test_cap.release()
                
            except Exception as e:
                print(f"   ❌ Camera {i}: Error - {e}")
                continue
        
        print(f"📹 Found {len(available_cameras)} available cameras: {available_cameras}")
        return available_cameras
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera with optimized settings for ultra-low latency capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            print(f"🚀 Initializing camera (device {self.device_id})...")
            
            # Detect available cameras first
            available_cameras = self.detect_available_cameras()
            if not available_cameras:
                print("❌ No cameras detected on system")
                return False
            
            # Use requested camera or fallback to first available
            if self.device_id not in available_cameras:
                print(f"⚠️  Requested camera {self.device_id} not available, using camera {available_cameras[0]}")
                self.device_id = available_cameras[0]
            
            # Initialize camera with optimal backend
            if self.backend == "auto":
                if platform.system() == "Windows":
                    self.camera = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                else:
                    self.camera = cv2.VideoCapture(self.device_id)
            else:
                backend_map = {
                    "dshow": cv2.CAP_DSHOW,
                    "v4l2": cv2.CAP_V4L2,
                    "gstreamer": cv2.CAP_GSTREAMER
                }
                backend_id = backend_map.get(self.backend, cv2.CAP_ANY)
                self.camera = cv2.VideoCapture(self.device_id, backend_id)
            
            if not self.camera.isOpened():
                print(f"❌ Failed to open camera device {self.device_id}")
                return False
            
            # Configure camera settings for optimal performance
            success = self._configure_camera_settings()
            if not success:
                print("⚠️  Some camera settings could not be applied")
            
            # Test frame capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("❌ Failed to capture test frame")
                return False
            
            self.is_initialized = True
            self._print_camera_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
    
    def _configure_camera_settings(self) -> bool:
        """Configure camera settings for optimal performance."""
        try:
            settings_applied = 0
            total_settings = 0
            
            # Core settings
            settings = [
                (cv2.CAP_PROP_FRAME_WIDTH, self.width),
                (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                (cv2.CAP_PROP_FPS, self.fps),
                (cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            ]
            
            # Optional settings with fallbacks
            optional_settings = [
                (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),  # Disable auto exposure
                (cv2.CAP_PROP_BRIGHTNESS, self.config.get("camera.brightness", 0.5)),
                (cv2.CAP_PROP_CONTRAST, self.config.get("camera.contrast", 0.5)),
                (cv2.CAP_PROP_SATURATION, self.config.get("camera.saturation", 0.5))
            ]
            
            # Apply core settings
            for prop, value in settings:
                total_settings += 1
                if self.camera.set(prop, value):
                    settings_applied += 1
                else:
                    print(f"⚠️  Could not set camera property {prop} to {value}")
            
            # Apply optional settings
            for prop, value in optional_settings:
                total_settings += 1
                if self.camera.set(prop, value):
                    settings_applied += 1
            
            # Set video codec if supported
            if hasattr(cv2, 'VideoWriter_fourcc'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
                    self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                    settings_applied += 1
                except:
                    pass
                total_settings += 1
            
            print(f"📊 Camera settings: {settings_applied}/{total_settings} applied successfully")
            return settings_applied >= len(settings)  # Core settings must succeed
            
        except Exception as e:
            print(f"❌ Error configuring camera settings: {e}")
            return False
    
    def _print_camera_info(self):
        """Print comprehensive camera information."""
        try:
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            backend_name = self.camera.getBackendName() if hasattr(self.camera, 'getBackendName') else "Unknown"
            
            print("✅ Camera initialized successfully:")
            print(f"   📐 Resolution: {actual_width}x{actual_height} (requested: {self.width}x{self.height})")
            print(f"   🎯 FPS: {actual_fps:.1f} (requested: {self.fps})")
            print(f"   🔧 Backend: {backend_name}")
            print(f"   📦 Buffer Size: {self.buffer_size}")
            print(f"   📹 Device ID: {self.device_id}")
            
        except Exception as e:
            print(f"⚠️  Could not display camera info: {e}")
    
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture frame from camera with performance optimizations and error handling.
        
        Returns:
            Tuple of (success flag, frame data)
        """
        if not self.is_initialized or not self.camera:
            return False, None
        
        capture_start = time.time()
        
        try:
            # Skip frames for performance optimization if needed
            if self.frame_skip_count > 0:
                for _ in range(self.frame_skip_count):
                    self.camera.grab()  # Fast frame grab without decode
                self.frame_skip_count = 0
            
            # Capture frame
            ret, frame = self.camera.read()
            
            if ret and frame is not None:
                self.last_frame = frame.copy()
                self.frame_count += 1
                
                # Record capture timing
                capture_time = time.time() - capture_start
                self.capture_times.append(capture_time)
                
                # Update FPS calculation
                self._update_fps_calculation()
                
                return True, frame
            else:
                self.dropped_frames += 1
                print(f"⚠️  Failed to capture frame (dropped: {self.dropped_frames})")
                
                # Return last known good frame if available
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
                return False, None
                
        except Exception as e:
            print(f"❌ Error capturing frame: {e}")
            self.dropped_frames += 1
            return False, None
    
    def _update_fps_calculation(self):
        """Update actual FPS calculation based on capture timing."""
        current_time = time.time()
        
        if current_time - self.last_fps_calculation >= 1.0:  # Update every second
            if len(self.capture_times) > 0:
                avg_capture_time = np.mean(list(self.capture_times)[-30:])  # Last 30 frames
                self.actual_fps = 1.0 / avg_capture_time if avg_capture_time > 0 else 0.0
                self.frame_rate_history.append(self.actual_fps)
            
            self.last_fps_calculation = current_time
    
    def set_frame_skip(self, skip_count: int):
        """
        Set number of frames to skip for performance optimization.
        
        Args:
            skip_count: Number of frames to skip on next capture
        """
        self.frame_skip_count = max(0, min(skip_count, self.frame_skip_threshold))
    
    def get_camera_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive camera statistics and performance metrics.
        
        Returns:
            Dictionary containing camera statistics
        """
        avg_capture_time = np.mean(self.capture_times) * 1000 if self.capture_times else 0.0
        
        return {
            'device_id': self.device_id,
            'resolution': f"{self.width}x{self.height}",
            'target_fps': self.fps,
            'actual_fps': self.actual_fps,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'drop_rate': (self.dropped_frames / max(self.frame_count + self.dropped_frames, 1)) * 100,
            'average_capture_time_ms': avg_capture_time,
            'buffer_size': self.buffer_size,
            'is_initialized': self.is_initialized,
            'backend': self.backend
        }
    
    def cleanup(self):
        """Clean up camera resources and release device."""
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
            self.is_initialized = False
            print("🧹 Camera resources released")
        except Exception as e:
            print(f"⚠️  Error during camera cleanup: {e}")

class ComprehensiveLoggingManager:
    """
    Military-grade logging system with performance tracking, error reporting,
    structured log formatting, and comprehensive debugging capabilities.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize comprehensive logging manager with advanced features.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        logging_config = config.get("logging", {})
        
        # Logging configuration
        self.log_file = logging_config.get("file_path", "gesture_control.log")
        self.log_level = getattr(logging, logging_config.get("level", "INFO").upper())
        self.max_file_size = logging_config.get("max_file_size_mb", 50) * 1024 * 1024
        self.backup_count = logging_config.get("backup_count", 5)
        self.console_output = logging_config.get("console_output", True)
        self.performance_logging = logging_config.get("performance_logging", True)
        self.gesture_logging = logging_config.get("gesture_logging", True)
        self.error_logging = logging_config.get("error_logging", True)
        
        # Setup logging system
        self._setup_comprehensive_logging()
        
        # Performance and event tracking
        self.gesture_events = []
        self.performance_snapshots = []
        self.error_events = []
        self.warning_events = []
        self.session_start_time = time.time()
        
        # Statistics
        self.total_gestures_logged = 0
        self.total_errors_logged = 0
        self.total_warnings_logged = 0
        
    def _setup_comprehensive_logging(self):
        """Configure advanced logging with multiple handlers and formatters."""
        # Create main logger
        self.logger = logging.getLogger("GestureControlSystem")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create detailed formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create simple formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"⚠️  Could not setup file logging: {e}")
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Performance logger (separate file)
        if self.performance_logging:
            try:
                perf_handler = RotatingFileHandler(
                    "performance.log",
                    maxBytes=self.max_file_size // 2,
                    backupCount=3,
                    encoding='utf-8'
                )
                perf_handler.setLevel(logging.INFO)
                perf_handler.setFormatter(detailed_formatter)
                
                self.perf_logger = logging.getLogger("PerformanceMonitor")
                self.perf_logger.addHandler(perf_handler)
                self.perf_logger.setLevel(logging.INFO)
            except Exception as e:
                print(f"⚠️  Could not setup performance logging: {e}")
                self.perf_logger = self.logger
        else:
            self.perf_logger = self.logger
        
        self.logger.info("📝 Comprehensive Logging System initialized")
        self.logger.info(f"   📁 Log file: {self.log_file}")
        self.logger.info(f"   📊 Log level: {logging.getLevelName(self.log_level)}")
        self.logger.info(f"   💾 Max file size: {self.max_file_size / (1024*1024):.1f} MB")
    
    def log_system_information(self):
        """Log comprehensive system information at startup."""
        self.logger.info("=" * 80)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 80)
        
        # Platform information
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Python Version: {platform.python_version()}")
        self.logger.info(f"OpenCV Version: {cv2.__version__}")
        
        try:
            self.logger.info(f"MediaPipe Version: {mp.__version__}")
        except:
            self.logger.info("MediaPipe Version: Unknown")
        
        # Hardware information
        if PSUTIL_AVAILABLE:
            self.logger.info(f"CPU Cores: {psutil.cpu_count()}")
            self.logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # GPU information
        if CUDA_AVAILABLE:
            self.logger.info(f"PyTorch Version: {torch.__version__}")
            self.logger.info(f"CUDA Available: Yes")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            self.logger.info("CUDA Available: No")
        
        # Library availability
        self.logger.info(f"PyInput Available: {PYNPUT_AVAILABLE}")
        self.logger.info(f"PyAutoGUI Available: {PYAUTOGUI_AVAILABLE}")
        self.logger.info(f"PSUtil Available: {PSUTIL_AVAILABLE}")
        
        self.logger.info("=" * 80)
        self.logger.info("SESSION START")
        self.logger.info("=" * 80)
    
    def log_gesture_detection(self, gesture: GestureType, confidence: float,
                            processing_time: float, actions: List[GameAction],
                            frame_number: int = 0):
        """
        Log comprehensive gesture detection event with all relevant data.
        
        Args:
            gesture: Detected gesture type
            confidence: Detection confidence score
            processing_time: Time taken for processing (seconds)
            actions: List of game actions executed
            frame_number: Frame number for tracking
        """
        if not self.gesture_logging:
            return
        
        timestamp = time.time()
        
        # Create detailed gesture event
        gesture_event = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'gesture': gesture.value,
            'confidence': confidence,
            'processing_time_ms': processing_time * 1000,
            'actions': [action.value for action in actions],
            'action_count': len(actions)
        }
        
        # Add to gesture events history
        self.gesture_events.append(gesture_event)
        self.total_gestures_logged += 1
        
        # Keep only recent events (last 1000)
        if len(self.gesture_events) > 1000:
            self.gesture_events = self.gesture_events[-1000:]
        
        # Log to file
        action_str = ", ".join([action.value for action in actions]) if actions else "none"
        self.logger.info(
            f"GESTURE | Frame:{frame_number:06d} | {gesture.value.upper()} | "
            f"Confidence:{confidence:.3f} | Time:{processing_time*1000:.1f}ms | "
            f"Actions:[{action_str}]"
        )
    
    def log_performance_metrics(self, stats: Dict[str, Any]):
        """
        Log comprehensive performance metrics with detailed analysis.
        
        Args:
            stats: Performance statistics dictionary
        """
        if not self.performance_logging:
            return
        
        timestamp = time.time()
        
        # Add timestamp to stats
        stats_with_time = {
            'timestamp': timestamp,
            **stats
        }
        
        # Add to performance snapshots
        self.performance_snapshots.append(stats_with_time)
        
        # Keep only recent snapshots (last 100)
        if len(self.performance_snapshots) > 100:
            self.performance_snapshots = self.performance_snapshots[-100:]
        
        # Log key performance metrics
        fps = stats.get('fps', {}).get('current', 0)
        latency = stats.get('timing', {}).get('total_latency_ms', 0)
        cpu = stats.get('hardware', {}).get('cpu_percent', 0)
        memory = stats.get('hardware', {}).get('memory_percent', 0)
        health = stats.get('quality', {}).get('system_health', 'UNKNOWN')
        
        self.perf_logger.info(
            f"PERFORMANCE | FPS:{fps:.1f} | Latency:{latency:.1f}ms | "
            f"CPU:{cpu:.1f}% | Memory:{memory:.1f}% | Health:{health}"
        )
        
        # Log alerts if any
        if 'alerts' in stats:
            for alert in stats['alerts']:
                self.logger.warning(f"PERFORMANCE ALERT | {alert}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None,
                  context: Optional[Dict[str, Any]] = None):
        """
        Log comprehensive error information with context and stack traces.
        
        Args:
            error_msg: Error message description
            exception: Optional exception object for detailed information
            context: Optional context dictionary with additional information
        """
        if not self.error_logging:
            return
        
        self.total_errors_logged += 1
        
        # Create error event
        error_event = {
            'timestamp': time.time(),
            'message': error_msg,
            'exception_type': type(exception).__name__ if exception else None,
            'exception_msg': str(exception) if exception else None,
            'context': context or {}
        }
        
        self.error_events.append(error_event)
        
        # Log error with context
        if context:
            context_str = " | ".join([f"{k}:{v}" for k, v in context.items()])
            error_msg = f"{error_msg} | Context: {context_str}"
        
        if exception:
            self.logger.error(f"ERROR | {error_msg} | Exception: {str(exception)}", exc_info=True)
        else:
            self.logger.error(f"ERROR | {error_msg}")
    
    def log_warning(self, warning_msg: str, context: Optional[Dict[str, Any]] = None):
        """
        Log warning message with optional context.
        
        Args:
            warning_msg: Warning message
            context: Optional context dictionary
        """
        self.total_warnings_logged += 1
        
        # Create warning event
        warning_event = {
            'timestamp': time.time(),
            'message': warning_msg,
            'context': context or {}
        }
        
        self.warning_events.append(warning_event)
        
        # Log warning with context
        if context:
            context_str = " | ".join([f"{k}:{v}" for k, v in context.items()])
            warning_msg = f"{warning_msg} | Context: {context_str}"
        
        self.logger.warning(f"WARNING | {warning_msg}")
    
    def log_camera_event(self, event_type: str, details: Dict[str, Any]):
        """Log camera-related events."""
        self.logger.info(f"CAMERA | {event_type} | {details}")
    
    def log_input_event(self, action: GameAction, success: bool, latency_ms: float):
        """Log game input events."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.debug(f"INPUT | {action.value} | {status} | {latency_ms:.1f}ms")
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive session summary with detailed statistics.
        
        Returns:
            Dictionary containing complete session analysis
        """
        session_duration = time.time() - self.session_start_time
        
        # Gesture analysis
        gesture_analysis = self._analyze_gesture_events()
        
        # Performance analysis
        performance_analysis = self._analyze_performance_snapshots()
        
        # Error analysis
        error_analysis = {
            'total_errors': self.total_errors_logged,
            'total_warnings': self.total_warnings_logged,
            'error_rate_per_hour': (self.total_errors_logged / (session_duration / 3600)) if session_duration > 0 else 0,
            'warning_rate_per_hour': (self.total_warnings_logged / (session_duration / 3600)) if session_duration > 0 else 0
        }
        
        summary = {
            'session_info': {
                'start_time': self.session_start_time,
                'duration_minutes': session_duration / 60,
                'duration_hours': session_duration / 3600
            },
            'gesture_analysis': gesture_analysis,
            'performance_analysis': performance_analysis,
            'error_analysis': error_analysis,
            'logging_stats': {
                'total_gestures_logged': self.total_gestures_logged,
                'total_errors_logged': self.total_errors_logged,
                'total_warnings_logged': self.total_warnings_logged,
                'log_file_size_mb': self._get_log_file_size()
            }
        }
        
        return summary
    
    def _analyze_gesture_events(self) -> Dict[str, Any]:
        """Analyze gesture events for summary statistics."""
        if not self.gesture_events:
            return {'total_gestures': 0}
        
        # Count gestures by type
        gesture_counts = {}
        confidence_scores = []
        processing_times = []
        
        for event in self.gesture_events:
            gesture = event['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            confidence_scores.append(event['confidence'])
            processing_times.append(event['processing_time_ms'])
        
        return {
            'total_gestures': len(self.gesture_events),
            'gesture_counts': gesture_counts,
            'average_confidence': np.mean(confidence_scores),
            'average_processing_time_ms': np.mean(processing_times),
            'gestures_per_minute': len(self.gesture_events) / ((time.time() - self.session_start_time) / 60)
        }
    
    def _analyze_performance_snapshots(self) -> Dict[str, Any]:
        """Analyze performance snapshots for summary statistics."""
        if not self.performance_snapshots:
            return {'snapshots_available': False}
        
        # Extract key metrics
        fps_values = []
        latency_values = []
        cpu_values = []
        memory_values = []
        
        for snapshot in self.performance_snapshots:
            if 'fps' in snapshot and 'current' in snapshot['fps']:
                fps_values.append(snapshot['fps']['current'])
            
            if 'timing' in snapshot and 'total_latency_ms' in snapshot['timing']:
                latency_values.append(snapshot['timing']['total_latency_ms'])
            
            if 'hardware' in snapshot:
                if 'cpu_percent' in snapshot['hardware']:
                    cpu_values.append(snapshot['hardware']['cpu_percent'])
                if 'memory_percent' in snapshot['hardware']:
                    memory_values.append(snapshot['hardware']['memory_percent'])
        
        return {
            'snapshots_available': True,
            'average_fps': np.mean(fps_values) if fps_values else 0,
            'average_latency_ms': np.mean(latency_values) if latency_values else 0,
            'average_cpu_percent': np.mean(cpu_values) if cpu_values else 0,
            'average_memory_percent': np.mean(memory_values) if memory_values else 0,
            'peak_fps': np.max(fps_values) if fps_values else 0,
            'peak_latency_ms': np.max(latency_values) if latency_values else 0
        }
    
    def _get_log_file_size(self) -> float:
        """Get current log file size in MB."""
        try:
            if os.path.exists(self.log_file):
                return os.path.getsize(self.log_file) / (1024 * 1024)
        except:
            pass
        return 0.0
    
    def cleanup(self):
        """Clean up logging resources and generate final summary."""
        # Generate and log session summary
        summary = self.generate_session_summary()
        
        self.logger.info("=" * 80)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 80)
        
        # Log session info
        session_info = summary['session_info']
        self.logger.info(f"Session Duration: {session_info['duration_minutes']:.1f} minutes")
        
        # Log gesture summary
        gesture_info = summary['gesture_analysis']
        if gesture_info.get('total_gestures', 0) > 0:
            self.logger.info(f"Total Gestures: {gesture_info['total_gestures']}")
            self.logger.info(f"Average Confidence: {gesture_info['average_confidence']:.3f}")
            self.logger.info(f"Gestures/min: {gesture_info['gestures_per_minute']:.1f}")
        
        # Log performance summary
        perf_info = summary['performance_analysis']
        if perf_info.get('snapshots_available', False):
            self.logger.info(f"Average FPS: {perf_info['average_fps']:.1f}")
            self.logger.info(f"Average Latency: {perf_info['average_latency_ms']:.1f}ms")
        
        # Log error summary
        error_info = summary['error_analysis']
        self.logger.info(f"Errors: {error_info['total_errors']}")
        self.logger.info(f"Warnings: {error_info['total_warnings']}")
        
        self.logger.info("=" * 80)
        self.logger.info("SESSION END")
        self.logger.info("=" * 80)
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

class MilitaryGradeGestureControlSystem:
    """
    Military-grade orchestration system that integrates all components for real-time
    hand gesture recognition and racing game control with ultra-low latency,
    comprehensive error handling, and advanced performance optimization.
    """
    
    def __init__(self):
        """Initialize the complete military-grade gesture control system."""
        print("🚀" * 30)
        print("MILITARY-GRADE REAL-TIME HAND GESTURE CONTROL SYSTEM")
        print("FOR RACING GAMES USING ADVANCED COMPUTER VISION")
        print("🚀" * 30)
        print("🏎️  Optimized for Forza Horizon 5 & All Racing Games")
        print("💻 Hardware: Intel i7-14700KF + RTX 4070 SUPER + 64GB DDR5")
        print("🎯 Target: <50ms Latency | 30+ FPS | 95%+ Accuracy")
        print("🚀" * 30)
        
        # System initialization timestamp
        self.system_start_time = time.time()
        
        # Initialize configuration system first
        print("🔧 Initializing configuration system...")
        self.config = ConfigurationManager()
        
        # Initialize logging system
        print("📝 Initializing comprehensive logging system...")
        self.logger = ComprehensiveLoggingManager(self.config)
        self.logger.log_system_information()
        
        # Initialize core components with comprehensive error handling
        self.components_initialized = False
        try:
            print("📹 Initializing camera management system...")
            self.camera_manager = CameraManager(self.config)
            
            print("🤖 Initializing hand gesture recognition system...")
            self.gesture_recognizer = HandGestureRecognizer(self.config)
            
            print("🎮 Initializing game input controller...")
            self.input_controller = GameInputController(self.config)
            
            print("📊 Initializing performance monitoring system...")
            self.performance_monitor = PerformanceMonitor()
            
            self.components_initialized = True
            
        except Exception as e:
            self.logger.log_error("Failed to initialize core components", e)
            raise
        
        # System state management
        self.system_running = False
        self.system_paused = False
        self.emergency_stop_active = False
        self.current_gesture = GestureType.NO_GESTURE
        self.current_confidence = 0.0
        self.last_gesture_time = time.time()
        
        # Threading for high-performance parallel processing
        self.frame_queue = queue.Queue(maxsize=20)
        self.gesture_queue = queue.Queue(maxsize=10)
        self.input_queue = queue.Queue(maxsize=5)
        
        # Thread management
        self.processing_threads = []
        self.shutdown_event = threading.Event()
        
        # Performance optimization parameters
        self.target_fps = self.config.get("performance.max_fps", 60)
        self.target_frame_time = 1.0 / self.target_fps
        self.target_latency_ms = self.config.get("performance.target_latency_ms", 25)
        
        # Statistics and monitoring
        self.frame_count = 0
        self.gesture_count = 0
        self.successful_inputs = 0
        self.failed_inputs = 0
        
        # Safety and emergency systems
        self._setup_emergency_systems()
        
        print("✅ Military-Grade Gesture Control System initialized successfully!")
        print(f"🔧 Configuration: {self.config.config_file}")
        print(f"📝 Logging: {self.logger.log_file}")
        print(f"🎯 Target Performance: {self.target_fps} FPS, <{self.target_latency_ms}ms latency")
    
    def _setup_emergency_systems(self):
        """Setup comprehensive emergency stop and safety systems."""
        def emergency_shutdown():
            """Critical emergency shutdown procedure."""
            print("\n🚨 EMERGENCY SHUTDOWN INITIATED!")
            self.emergency_stop_active = True
            self.system_running = False
            
            # Emergency brake activation
            if hasattr(self, 'input_controller') and self.input_controller.is_available:
                self.input_controller.emergency_stop()
            
            # Signal all threads to stop
            self.shutdown_event.set()
            
            self.logger.log_error("Emergency shutdown activated")
        
        # Setup emergency key listener
        def on_emergency_key(key):
            try:
                emergency_keys = [Key.esc, Key.f12]  # ESC or F12 for emergency stop
                if key in emergency_keys:
                    emergency_shutdown()
                elif hasattr(key, 'char') and key.char and key.char.lower() == 'q':
                    emergency_shutdown()
            except AttributeError:
                pass
        
        # Start emergency key listener
        if PYNPUT_AVAILABLE:
            try:
                self.emergency_listener = keyboard.Listener(on_press=on_emergency_key)
                self.emergency_listener.start()
                print("🚨 Emergency stop system active (Press ESC, F12, or Q to stop)")
            except Exception as e:
                print(f"⚠️  Could not start emergency listener: {e}")
        
        # Signal handler for Ctrl+C
        def signal_handler(signum, frame):
            emergency_shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_system(self) -> bool:
        """
        Initialize all system components with military-grade error handling and validation.
        
        Returns:
            True if all components initialized successfully, False otherwise
        """
        if not self.components_initialized:
            self.logger.log_error("Core components not initialized")
            return False
        
        try:
            print("\n🔍 Running comprehensive system initialization...")
            
            # Initialize camera with device detection
            print("📹 Initializing camera system...")
            if not self.camera_manager.initialize_camera():
                self.logger.log_error("Camera initialization failed")
                return False
            
            # Test camera capture
            print("📸 Testing camera capture...")
            ret, test_frame = self.camera_manager.capture_frame()
            if not ret or test_frame is None:
                self.logger.log_error("Camera capture test failed")
                return False
            
            # Test gesture recognition system
            print("🤖 Testing gesture recognition...")
            hands, processed_frame = self.gesture_recognizer.process_frame(test_frame)
            if hands is None:
                self.logger.log_error("Gesture recognition test failed")
                return False
            
            # Test input controller
            print("🎮 Testing input controller...")
            if not self.input_controller.is_available:
                self.logger.log_error("Input controller not available")
                return False
            
            # Validate configuration
            print("🔧 Validating system configuration...")
            if not self._validate_system_configuration():
                self.logger.log_error("System configuration validation failed")
                return False
            
            print("✅ System initialization completed successfully!")
            print("🎯 All systems operational and ready for gesture control")
            
            return True
            
        except Exception as e:
            self.logger.log_error("Critical error during system initialization", e)
            return False
    
    def _validate_system_configuration(self) -> bool:
        """Validate system configuration for optimal performance."""
        validation_errors = []
        
        # Check camera configuration
        camera_stats = self.camera_manager.get_camera_statistics()
        if camera_stats.get('actual_fps', 0) < 20:
            validation_errors.append("Camera FPS too low for optimal performance")
        
        # Check MediaPipe configuration
        if self.gesture_recognizer.confidence_threshold < 0.5:
            validation_errors.append("Gesture confidence threshold too low")
        
        # Check performance targets
        if self.target_fps < 25:
            validation_errors.append("Target FPS too low for gaming")
        
        if validation_errors:
            for error in validation_errors:
                self.logger.log_warning(f"Configuration validation: {error}")
            return len(validation_errors) < 3  # Allow minor issues
        
        return True
    
    def start_processing_threads(self):
        """Start all background processing threads for parallel execution."""
        print("🧵 Starting high-performance processing threads...")
        
        # Frame processing thread (highest priority)
        frame_thread = threading.Thread(
            target=self._frame_processing_loop,
            name="FrameProcessor",
            daemon=True
        )
        
        # Gesture classification thread
        gesture_thread = threading.Thread(
            target=self._gesture_processing_loop,
            name="GestureProcessor",
            daemon=True
        )
        
        # Input execution thread (ultra-low latency)
        input_thread = threading.Thread(
            target=self._input_processing_loop,
            name="InputProcessor",
            daemon=True
        )
        
        # Performance monitoring thread
        monitor_thread = threading.Thread(
            target=self._performance_monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        
        # Start all threads
        threads = [frame_thread, gesture_thread, input_thread, monitor_thread]
        for thread in threads:
            thread.start()
            self.processing_threads.append(thread)
        
        print(f"✅ Started {len(threads)} processing threads")
        
        # Verify threads are running
        time.sleep(0.1)
        running_threads = sum(1 for t in threads if t.is_alive())
        print(f"🔍 Thread status: {running_threads}/{len(threads)} threads active")
        
        return running_threads == len(threads)
    
    def _frame_processing_loop(self):
        """High-priority frame capture and initial processing loop."""
        thread_name = threading.current_thread().name
        self.logger.logger.debug(f"Thread {thread_name} started")
        
        frame_count = 0
        last_fps_report = time.time()
        
        while self.system_running and not self.shutdown_event.is_set():
            try:
                frame_start = time.time()
                
                # Capture frame from camera
                ret, frame = self.camera_manager.capture_frame()
                if not ret or frame is None:
                    self.logger.log_warning("Frame capture failed", {'frame_count': frame_count})
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                self.frame_count = frame_count
                
                # Add frame to processing queue if not full
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait((frame, frame_count, frame_start))
                    else:
                        # Drop frame if queue is full (performance optimization)
                        self.logger.log_warning("Frame queue full, dropping frame")
                except queue.Full:
                    pass
                
                # Frame rate control
                frame_time = time.time() - frame_start
                self.performance_monitor.record_frame_time(frame_time)
                
                # Report FPS periodically
                if time.time() - last_fps_report >= 5.0:
                    stats = self.performance_monitor.get_comprehensive_stats()
                    current_fps = stats['fps']['current']
                    self.logger.logger.debug(f"{thread_name}: Processing at {current_fps:.1f} FPS")
                    last_fps_report = time.time()
                
                # Adaptive frame rate control
                if frame_time < self.target_frame_time:
                    time.sleep(self.target_frame_time - frame_time)
                
            except Exception as e:
                self.logger.log_error(f"Error in {thread_name}", e)
                time.sleep(0.01)
        
        self.logger.logger.debug(f"Thread {thread_name} stopped")
    
    def _gesture_processing_loop(self):
        """Gesture detection and classification processing loop."""
        thread_name = threading.current_thread().name
        self.logger.logger.debug(f"Thread {thread_name} started")
        
        while self.system_running and not self.shutdown_event.is_set():
            try:
                # Get frame from queue
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                    frame, frame_number, capture_time = frame_data
                except queue.Empty:
                    continue
                
                processing_start = time.time()
                
                # Process frame for hand detection
                detection_start = time.time()
                hands, annotated_frame = self.gesture_recognizer.process_frame(frame)
                detection_time = time.time() - detection_start
                self.performance_monitor.record_detection_time(detection_time)
                
                # Classify gesture
                classification_start = time.time()
                gesture, confidence = self.gesture_recognizer.classify_gesture(hands)
                classification_time = time.time() - classification_start
                self.performance_monitor.record_classification_time(classification_time)
                
                # Update current gesture state
                self.current_gesture = gesture
                self.current_confidence = confidence
                self.last_gesture_time = time.time()
                self.gesture_count += 1
                
                # Calculate total processing time
                total_processing_time = time.time() - processing_start
                self.performance_monitor.record_processing_time(total_processing_time)
                
                # Create gesture frame data
                gesture_frame = GestureFrame(
                    timestamp=time.time(),
                    left_hand=next((h for h in hands if h.handedness == "Left"), None),
                    right_hand=next((h for h in hands if h.handedness == "Right"), None),
                    gesture_type=gesture,
                    confidence=confidence,
                    frame_number=frame_number,
                    processing_time_ms=total_processing_time * 1000,
                    camera_fps=self.performance_monitor.current_fps,
                    system_latency_ms=(time.time() - capture_time) * 1000
                )
                
                # Add to input processing queue
                try:
                    if not self.gesture_queue.full():
                        self.gesture_queue.put_nowait((gesture_frame, annotated_frame))
                except queue.Full:
                    self.logger.log_warning("Gesture queue full")
                
                # Check for latency violations
                total_latency = (time.time() - capture_time) * 1000
                if total_latency > self.target_latency_ms:
                    self.performance_monitor.record_latency_violation()
                
            except Exception as e:
                self.logger.log_error(f"Error in {thread_name}", e)
                time.sleep(0.001)
        
        self.logger.logger.debug(f"Thread {thread_name} stopped")
    
    def _input_processing_loop(self):
        """Ultra-low latency game input processing loop."""
        thread_name = threading.current_thread().name
        self.logger.logger.debug(f"Thread {thread_name} started")
        
        while self.system_running and not self.shutdown_event.is_set():
            try:
                # Get gesture from queue
                try:
                    gesture_data = self.gesture_queue.get(timeout=0.1)
                    gesture_frame, display_frame = gesture_data
                except queue.Empty:
                    continue
                
                # Execute game input with minimal latency
                input_start = time.time()
                
                executed_actions = self.input_controller.execute_gesture_action(
                    gesture_frame.gesture_type
                )
                
                input_time = time.time() - input_start
                self.performance_monitor.record_input_time(input_time)
                
                # Update statistics
                if executed_actions:
                    self.successful_inputs += 1
                else:
                    self.failed_inputs += 1
                
                # Log gesture detection
                self.logger.log_gesture_detection(
                    gesture_frame.gesture_type,
                    gesture_frame.confidence,
                    gesture_frame.processing_time_ms / 1000,
                    executed_actions,
                    gesture_frame.frame_number
                )
                
                # Display frame if enabled
                if self.config.get("display.show_camera_feed", True) and display_frame is not None:
                    self._update_display_frame(display_frame, gesture_frame, executed_actions)
                
            except Exception as e:
                self.logger.log_error(f"Error in {thread_name}", e)
                self.failed_inputs += 1
                time.sleep(0.001)
        
        self.logger.logger.debug(f"Thread {thread_name} stopped")
    
    def _performance_monitoring_loop(self):
        """Performance monitoring and statistics collection loop."""
        thread_name = threading.current_thread().name
        self.logger.logger.debug(f"Thread {thread_name} started")
        
        last_stats_time = time.time()
        stats_interval = 10.0  # Log detailed stats every 10 seconds
        
        while self.system_running and not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Collect comprehensive statistics
                if current_time - last_stats_time >= stats_interval:
                    stats = self._collect_comprehensive_statistics()
                    
                    # Log performance metrics
                    self.logger.log_performance_metrics(stats)
                    
                    # Check for performance alerts
                    alerts = self.performance_monitor.get_performance_alerts()
                    if alerts:
                        for alert in alerts:
                            self.logger.log_warning(f"Performance Alert: {alert}")
                    
                    last_stats_time = current_time
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                self.logger.log_error(f"Error in {thread_name}", e)
                time.sleep(1.0)
        
        self.logger.logger.debug(f"Thread {thread_name} stopped")
    
    def _collect_comprehensive_statistics(self) -> Dict[str, Any]:
        """Collect comprehensive system statistics from all components."""
        stats = self.performance_monitor.get_comprehensive_stats()
        
        # Add component-specific statistics
        stats['camera'] = self.camera_manager.get_camera_statistics()
        stats['gesture_recognition'] = self.gesture_recognizer.get_recognition_statistics()
        stats['input_controller'] = self.input_controller.get_input_statistics()
        
        # Add system-level statistics
        stats['system'] = {
            'uptime_seconds': time.time() - self.system_start_time,
            'frame_count': self.frame_count,
            'gesture_count': self.gesture_count,
            'successful_inputs': self.successful_inputs,
            'failed_inputs': self.failed_inputs,
            'input_success_rate': (self.successful_inputs / max(self.successful_inputs + self.failed_inputs, 1)) * 100,
            'current_gesture': self.current_gesture.value,
            'current_confidence': self.current_confidence,
            'emergency_stop_active': self.emergency_stop_active
        }
        
        # Add performance alerts
        stats['alerts'] = self.performance_monitor.get_performance_alerts()
        
        return stats
    
    def _update_display_frame(self, frame: np.ndarray, gesture_frame: GestureFrame, 
                             actions: List[GameAction]):
        """Update display frame with comprehensive overlay information."""
        try:
            # Create enhanced display frame
            display_frame = frame.copy()
            
            # Add gesture information
            if self.config.get("display.show_gesture_info", True):
                self._add_gesture_overlay(display_frame, gesture_frame, actions)
            
            # Add performance information
            if self.config.get("display.show_performance_stats", True):
                self._add_performance_overlay(display_frame)
            
            # Add system status
            self._add_system_status_overlay(display_frame)
            
            # Display the frame
            cv2.imshow("Military-Grade Gesture Control", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._handle_display_keyboard_input(key)
            
        except Exception as e:
            self.logger.log_error("Error updating display frame", e)
    
    def _add_gesture_overlay(self, frame: np.ndarray, gesture_frame: GestureFrame, 
                           actions: List[GameAction]):
        """Add gesture information overlay to display frame."""
        try:
            # Colors from configuration
            text_color = tuple(self.config.get("display.text_color", [0, 255, 0]))
            font_scale = self.config.get("display.text_size", 0.7)
            
            # Current gesture
            gesture_text = f"GESTURE: {gesture_frame.gesture_type.value.replace('_', ' ').upper()}"
            cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, text_color, 2)
            
            # Confidence
            confidence_text = f"CONFIDENCE: {gesture_frame.confidence:.3f}"
            cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale * 0.8, text_color, 2)
            
            # Actions
            if actions:
                action_text = f"ACTIONS: {', '.join([a.value.upper() for a in actions])}"
                cv2.putText(frame, action_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale * 0.7, text_color, 2)
            
            # Hand count
            hand_count = gesture_frame.get_hand_count()
            hand_text = f"HANDS: {hand_count}/2"
            cv2.putText(frame, hand_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale * 0.7, text_color, 2)
            
        except Exception as e:
            self.logger.log_error("Error adding gesture overlay", e)
    
    def _add_performance_overlay(self, frame: np.ndarray):
        """Add performance information overlay to display frame."""
        try:
            stats = self.performance_monitor.get_comprehensive_stats()
            text_color = tuple(self.config.get("display.text_color", [0, 255, 0]))
            font_scale = self.config.get("display.text_size", 0.7) * 0.6
            
            h, w = frame.shape[:2]
            x_offset = w - 250
            
            # FPS
            fps = stats['fps']['current']
            fps_color = text_color if fps >= 25 else (0, 165, 255)  # Orange if low FPS
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, fps_color, 2)
            
            # Latency
            latency = stats['timing']['total_latency_ms']
            latency_color = text_color if latency < 50 else (0, 165, 255)  # Orange if high
            latency_text = f"LATENCY: {latency:.1f}ms"
            cv2.putText(frame, latency_text, (x_offset, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, latency_color, 2)
            
            # System health
            health = stats['quality']['system_health']
            health_colors = {
                'EXCELLENT': (0, 255, 0),    # Green
                'GOOD': (0, 255, 255),       # Yellow
                'FAIR': (0, 165, 255),       # Orange
                'POOR': (0, 100, 255),       # Red-orange
                'CRITICAL': (0, 0, 255)      # Red
            }
            health_color = health_colors.get(health, text_color)
            health_text = f"HEALTH: {health}"
            cv2.putText(frame, health_text, (x_offset, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, health_color, 2)
            
            # CPU and Memory
            cpu = stats['hardware']['cpu_percent']
            memory = stats['hardware']['memory_percent']
            system_text = f"CPU: {cpu:.0f}% | MEM: {memory:.0f}%"
            cv2.putText(frame, system_text, (x_offset, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale * 0.8, text_color, 1)
            
        except Exception as e:
            self.logger.log_error("Error adding performance overlay", e)
    
    def _add_system_status_overlay(self, frame: np.ndarray):
        """Add system status overlay to display frame."""
        try:
            text_color = tuple(self.config.get("display.text_color", [0, 255, 0]))
            font_scale = self.config.get("display.text_size", 0.7) * 0.5
            
            h, w = frame.shape[:2]
            
            # System status
            status_text = "SYSTEM: OPERATIONAL" if not self.emergency_stop_active else "SYSTEM: EMERGENCY STOP"
            status_color = text_color if not self.emergency_stop_active else (0, 0, 255)
            cv2.putText(frame, status_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, status_color, 2)
            
            # Controls reminder
            controls_text = "ESC: Emergency Stop | P: Pause | Q: Quit"
            cv2.putText(frame, controls_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (200, 200, 200), 1)
            
            # Frame counter
            frame_text = f"FRAME: {self.frame_count:06d}"
            cv2.putText(frame, frame_text, (w - 150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, text_color, 1)
            
        except Exception as e:
            self.logger.log_error("Error adding system status overlay", e)
    
    def _handle_display_keyboard_input(self, key: int):
        """Handle keyboard input from display window."""
        if key == 27:  # ESC
            print("ESC pressed - initiating emergency stop...")
            self.emergency_stop_active = True
            self.system_running = False
        elif key == ord('p') or key == ord('P'):
            self.system_paused = not self.system_paused
            status = "PAUSED" if self.system_paused else "RESUMED"
            print(f"System {status}")
            self.logger.logger.info(f"System {status} by user")
        elif key == ord('q') or key == ord('Q'):
            print("Q pressed - shutting down system...")
            self.system_running = False
    
    def print_comprehensive_gesture_guide(self):
        """Print comprehensive gesture guide with advanced features."""
        print("\n" + "🎮" * 25)
        print("COMPREHENSIVE GESTURE CONTROL GUIDE")
        print("Military-Grade Precision Racing Control")
        print("🎮" * 25)
        
        print("\n📍 OPTIMAL SETUP:")
        print("   • Distance: 2-3 feet from webcam")
        print("   • Lighting: Bright, even lighting on hands")
        print("   • Background: Plain, non-moving background")
        print("   • Positioning: Both hands visible in camera frame")
        print("   • Stability: Hold gestures steady for 2-3 frames")
        
        print("\n🤲 PRIMARY GESTURES:")
        print("   🛑 E-BRAKE (Emergency Brake):")
        print("      • Both hands OPEN with fingers spread")
        print("      • Hands aligned horizontally")
        print("      • Game Action: Emergency brake (Spacebar)")
        print("      • Use: Immediate stop, holding position")
        
        print("   🚗 ACCELERATE (Move Forward):")
        print("      • Both hands CLOSED (fists)")
        print("      • Hands aligned horizontally")
        print("      • Game Action: Accelerate forward (W key)")
        print("      • Use: Forward movement, speed control")
        
        print("   🔄 REVERSE (Move Backward):")
        print("      • Right hand OPEN, left hand CLOSED")
        print("      • Hands aligned horizontally")
        print("      • Game Action: Reverse gear (S key)")
        print("      • Use: Backward movement, backing up")
        
        print("   🔧 START/IDLE:")
        print("      • Right hand CLOSED, left hand OPEN")
        print("      • Hands aligned horizontally")
        print("      • Game Action: Engine start/idle (E key)")
        print("      • Use: Engine preparation, idle state")
        
        print("\n🌟 REVOLUTIONARY COMBINED CONTROLS:")
        print("   🏎️ ACCELERATE + STEER LEFT:")
        print("      • Both hands FISTS + left hand HIGHER")
        print("      • Game Action: W + A keys simultaneously")
        print("      • Use: Smooth left turns while accelerating")
        
        print("   🏎️ ACCELERATE + STEER RIGHT:")
        print("      • Both hands FISTS + right hand HIGHER")
        print("      • Game Action: W + D keys simultaneously")
        print("      • Use: Smooth right turns while accelerating")
        
        print("   🔄 REVERSE + STEER:")
        print("      • Reverse gesture + height difference")
        print("      • Game Action: S + (A or D) keys")
        print("      • Use: Reverse while turning")
        
        print("   🛑 E-BRAKE + STEER:")
        print("      • E-brake gesture + height difference")
        print("      • Game Action: Spacebar + (A or D) keys")
        print("      • Use: Emergency maneuvers, drifting")
        
        print("\n⚡ PERFORMANCE TARGETS:")
        print(f"   • Latency: <{self.target_latency_ms}ms gesture-to-action")
        print(f"   • FPS: {self.target_fps}+ frames per second")
        print("   • Accuracy: 95%+ gesture recognition")
        print("   • Stability: 3-frame gesture consistency")
        
        print("\n🚨 SAFETY & EMERGENCY CONTROLS:")
        print("   • ESC: Instant emergency stop (releases all keys)")
        print("   • F12: Alternative emergency stop")
        print("   • Q: Quit system gracefully")
        print("   • P: Pause/resume gesture recognition")
        print("   • Auto-brake: Activated when hands not detected")
        
        print("\n🎯 GAMING OPTIMIZATION TIPS:")
        print("   • Practice basic gestures before racing")
        print("   • Start with simple tracks to learn system")
        print("   • Use combined controls for advanced maneuvers")
        print("   • Monitor performance stats for optimal setup")
        print("   • Ensure stable camera mounting")
        print("   • Close unnecessary applications for performance")
        
        print("🎮" * 25)
        print("Ready for Military-Grade Gesture Racing! 🏁")
        print("🎮" * 25)
    
    def run_main_system_loop(self):
        """
        Execute the main system loop with comprehensive error handling and monitoring.
        
        Returns:
            True if system completed successfully, False if errors occurred
        """
        try:
            print("\n🚀 Starting main system loop...")
            print("📊 Monitoring performance in real-time...")
            print("🎯 Target: Ultra-low latency gesture recognition")
            print("-" * 60)
            
            # Start background processing threads
            if not self.start_processing_threads():
                self.logger.log_error("Failed to start processing threads")
                return False
            
            # Main monitoring loop
            loop_count = 0
            last_status_update = time.time()
            status_interval = 30.0  # Status update every 30 seconds
            
            while self.system_running and not self.emergency_stop_active and not self.shutdown_event.is_set():
                loop_start = time.time()
                loop_count += 1
                
                # System health monitoring
                if time.time() - last_status_update >= status_interval:
                    self._print_system_status()
                    last_status_update = time.time()
                
                # Check for critical system issues
                if not self._check_system_health():
                    self.logger.log_error("Critical system health issue detected")
                    break
                
                # Main loop timing control
                loop_time = time.time() - loop_start
                if loop_time < 0.1:  # 10Hz main loop
                    time.sleep(0.1 - loop_time)
            
            print("\n🛑 Main system loop stopped")
            return not self.emergency_stop_active
            
        except KeyboardInterrupt:
            print("\n⌨️  Keyboard interrupt received")
            return False
        except Exception as e:
            self.logger.log_error("Critical error in main system loop", e)
            return False
    
    def _print_system_status(self):
        """Print comprehensive system status update."""
        try:
            stats = self._collect_comprehensive_statistics()
            
            print("\n" + "📊" * 20)
            print("SYSTEM STATUS UPDATE")
            print("📊" * 20)
            
            # Performance metrics
            fps = stats['fps']['current']
            latency = stats['timing']['total_latency_ms']
            health = stats['quality']['system_health']
            
            print(f"🎯 Performance: {fps:.1f} FPS | {latency:.1f}ms latency | {health}")
            
            # System counters
            print(f"📈 Processed: {self.frame_count} frames | {self.gesture_count} gestures")
            print(f"🎮 Input Success: {stats['system']['input_success_rate']:.1f}%")
            
            # Current gesture
            print(f"✋ Current: {self.current_gesture.value.replace('_', ' ').upper()} ({self.current_confidence:.3f})")
            
            # Hardware utilization
            cpu = stats['hardware']['cpu_percent']
            memory = stats['hardware']['memory_percent']
            print(f"💻 Hardware: CPU {cpu:.1f}% | Memory {memory:.1f}%")
            
            # Alerts
            alerts = stats.get('alerts', [])
            if alerts:
                print(f"⚠️  Alerts: {len(alerts)} active")
                for alert in alerts[:3]:  # Show first 3 alerts
                    print(f"   • {alert}")
            else:
                print("✅ No performance alerts")
            
            print("📊" * 20)
            
        except Exception as e:
            self.logger.log_error("Error printing system status", e)
    
    def _check_system_health(self) -> bool:
        """
        Check critical system health indicators.
        
        Returns:
            True if system is healthy, False if critical issues detected
        """
        try:
            # Check thread health
            alive_threads = sum(1 for t in self.processing_threads if t.is_alive())
            if alive_threads < len(self.processing_threads):
                self.logger.log_error(f"Thread failure: {alive_threads}/{len(self.processing_threads)} alive")
                return False
            
            # Check queue health
            if self.frame_queue.qsize() >= self.frame_queue.maxsize:
                self.logger.log_warning("Frame queue consistently full")
            
            # Check camera health
            camera_stats = self.camera_manager.get_camera_statistics()
            if camera_stats.get('drop_rate', 0) > 20:  # >20% drop rate
                self.logger.log_warning(f"High camera drop rate: {camera_stats['drop_rate']:.1f}%")
            
            # Check input controller health
            if not self.input_controller.is_available:
                self.logger.log_error("Input controller not available")
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error("Error checking system health", e)
            return False
    
    def run(self) -> bool:
        """
        Main entry point to run the complete military-grade gesture control system.
        
        Returns:
            True if system completed successfully, False if errors occurred
        """
        try:
            # Print comprehensive gesture guide
            self.print_comprehensive_gesture_guide()
            
            # Initialize all system components
            if not self.initialize_system():
                print("❌ System initialization failed. Cannot continue.")
                return False
            
            # Wait for user confirmation
            print("\n🏁 System ready for gesture control!")
            print("📹 Camera active and gesture recognition loaded")
            print("🎮 Game input controller ready")
            input("\n👋 Position your hands and press ENTER to start gesture control...")
            
            # Start the main system
            print("\n🚀 Activating military-grade gesture control...")
            self.system_running = True
            
            success = self.run_main_system_loop()
            
            if success and not self.emergency_stop_active:
                print("\n✅ System completed successfully!")
            else:
                print("\n⚠️  System stopped due to errors or emergency stop")
            
            return success
            
        except Exception as e:
            self.logger.log_error("Critical error in main run method", e)
            print(f"\n❌ Critical system error: {e}")
            return False
        
        finally:
            self.cleanup_system()
    
    def cleanup_system(self):
        """
        Comprehensive system cleanup with resource deallocation and final reporting.
        """
        print("\n🧹 Initiating comprehensive system cleanup...")
        
        # Stop main system
        self.system_running = False
        self.shutdown_event.set()
        
        # Emergency stop to release all inputs
        if hasattr(self, 'input_controller') and self.input_controller.is_available:
            self.input_controller.emergency_stop()
            time.sleep(0.1)  # Allow input to process
            self.input_controller.cleanup()
        
        # Stop processing threads
        print("🧵 Stopping processing threads...")
        if hasattr(self, 'processing_threads'):
            for thread in self.processing_threads:
                if thread.is_alive():
                    thread.join(timeout=2.0)
        
        # Cleanup core components
        if hasattr(self, 'camera_manager'):
            self.camera_manager.cleanup()
        
        if hasattr(self, 'gesture_recognizer'):
            self.gesture_recognizer.cleanup()
        
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.cleanup()
        
        # Stop emergency listener
        if hasattr(self, 'emergency_listener'):
            try:
                self.emergency_listener.stop()
            except:
                pass
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Final logging and cleanup
        if hasattr(self, 'logger'):
            print("📝 Generating final session report...")
            self.logger.cleanup()
        
        # Calculate final statistics
        total_time = time.time() - self.system_start_time
        print(f"\n📊 Final Statistics:")
        print(f"   ⏱️  Total Runtime: {total_time:.1f} seconds")
        print(f"   🖼️  Frames Processed: {self.frame_count}")
        print(f"   ✋ Gestures Detected: {self.gesture_count}")
        print(f"   🎮 Successful Inputs: {self.successful_inputs}")
        
        if self.gesture_count > 0:
            print(f"   📈 Average Processing Rate: {self.gesture_count / total_time:.1f} gestures/sec")
        
        print("\n✅ System cleanup completed successfully!")
        print("🏁 Thank you for using the My Gesture Control System!")
        print("🎮 Ready for your next racing adventure!")

def main():
    """
    Main entry point for the Military-Grade Real-Time Hand Gesture Control System.
    Handles system initialization, execution, and graceful shutdown with comprehensive error handling.
    """
    print("🎮" * 30)
    print("MILITARY-GRADE GESTURE CONTROL SYSTEM")
    print("Real-Time Hand Gesture Control for Racing Games")
    print("Author: Muhammad Kashan Tariq | Version: 1.0.0")
    print("Optimized for Intel i7-14700KF + RTX 4070 SUPER + 64GB DDR5")
    print("🎮" * 30)
    
    # System requirements validation
    print("\n🔍 Performing comprehensive system validation...")
    
    # Python version check
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print(f"❌ ERROR: Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        print("Please upgrade Python and try again.")
        return 1
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Critical library validation
    required_libraries = {
        'cv2': ('OpenCV', 'Computer vision processing'),
        'mediapipe': ('MediaPipe', 'Hand landmark detection'),
        'numpy': ('NumPy', 'Numerical computations'),
        'pynput': ('PyInput', 'Game input simulation')
    }
    
    missing_libraries = []
    for lib, (name, description) in required_libraries.items():
        try:
            __import__(lib)
            print(f"✅ {name}: Available ({description})")
        except ImportError:
            missing_libraries.append(name)
            print(f"❌ {name}: MISSING - {description}")
    
    if missing_libraries:
        print(f"\n❌ CRITICAL ERROR: Missing required libraries: {', '.join(missing_libraries)}")
        print("Please install missing libraries using:")
        print("   pip install -r requirements.txt")
        print("Then restart the system.")
        return 1
    
    # Hardware validation
    print(f"\n💻 Hardware Validation:")
    if PSUTIL_AVAILABLE:
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   🔥 CPU Cores: {cpu_count}")
        print(f"   🧠 System Memory: {memory_gb:.1f} GB")
        
        if cpu_count < 4:
            print("   ⚠️  WARNING: Low CPU core count may affect performance")
        if memory_gb < 8:
            print("   ⚠️  WARNING: Low memory may affect performance")
    
    # GPU validation
    if CUDA_AVAILABLE:
        print(f"   🚀 GPU: {CUDA_DEVICE_NAME}")
        print(f"   🎯 CUDA Memory: {CUDA_MEMORY_GB:.1f} GB")
        print("   ✅ GPU acceleration available")
    else:
        print("   ⚠️  GPU acceleration not available - using CPU only")
        print("   📝 Note: Performance may be reduced without GPU")
    
    print("\n✅ System validation completed successfully!")
    print("🚀 All requirements met for optimal performance")
    
    # Main system execution
    try:
        print("\n🏗️  Initializing Military-Grade Gesture Control System...")
        
        # Create and run the gesture control system
        system = MilitaryGradeGestureControlSystem()
        success = system.run()
        
        # Determine exit status
        if success:
            print("\n🎉 Mission accomplished! System completed successfully!")
            print("🏁 Ready for your next racing session!")
            return 0
        else:
            print("\n⚠️  System encountered issues during execution.")
            print("📝 Check logs for detailed information.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⌨️  System interrupted by user (Ctrl+C)")
        print("🛑 Performing emergency shutdown...")
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        print(f"\n💥 CRITICAL SYSTEM ERROR: {e}")
        print("📝 Please check system requirements and try again.")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    """
    Entry point for the Military-Grade Real-Time Hand Gesture Control System
    with comprehensive error handling and system requirements validation.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💀 FATAL ERROR: {e}")
        print("🆘 System cannot continue - please contact support")
        sys.exit(255)
