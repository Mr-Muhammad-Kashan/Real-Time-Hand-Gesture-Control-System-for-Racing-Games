#!/usr/bin/env python3
"""
================================================================
System Test Script for Military-Grade Hand Gesture Control System
================================================================

Quick system validation and compatibility testing script.
Tests all major components before running the main system.

Author: Muhammad Kashan Tariq
Version: 1.0.0
================================================================
"""

import sys
import time
import traceback

def test_imports():
    """Test all required imports and report versions."""
    print("🔍 Testing Python library imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: Failed to import - {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe: Failed to import - {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: Failed to import - {e}")
        return False
    
    try:
        import pynput
        print(f"✅ PyInput: Available")
    except ImportError as e:
        print(f"❌ PyInput: Failed to import - {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA: Not available")
    except ImportError as e:
        print(f"⚠️  PyTorch: Not available - {e}")
    
    try:
        import psutil
        print(f"✅ PSUtil: Available")
    except ImportError as e:
        print(f"⚠️  PSUtil: Not available - {e}")
    
    return True

def test_mediapipe_compatibility():
    """Test MediaPipe Hands initialization with version compatibility."""
    print("\n🤖 Testing MediaPipe Hands compatibility...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_hands = mp.solutions.hands
        
        # Test configuration with version compatibility
        hands_config = {
            'static_image_mode': False,
            'max_num_hands': 2,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5
        }
        
        # Try with model_complexity first
        try:
            hands_config['model_complexity'] = 1
            hands = mp_hands.Hands(**hands_config)
            print("✅ MediaPipe Hands initialized WITH model_complexity")
        except TypeError as e:
            if 'model_complexity' in str(e):
                # Remove model_complexity for newer versions
                hands_config.pop('model_complexity', None)
                hands = mp_hands.Hands(**hands_config)
                print("✅ MediaPipe Hands initialized WITHOUT model_complexity (newer version)")
            else:
                raise e
        
        # Test basic processing
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = hands.process(test_frame)
        print("✅ MediaPipe processing test successful")
        
        hands.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        traceback.print_exc()
        return False

def test_camera_detection():
    """Test camera device detection."""
    print("\n📹 Testing camera detection...")
    
    try:
        import cv2
        import platform
        
        available_cameras = []
        
        # Test camera indices 0-3
        for i in range(4):
            try:
                if platform.system() == "Windows":
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        print(f"✅ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
                    else:
                        print(f"⚠️  Camera {i}: Opens but cannot read frames")
                else:
                    print(f"❌ Camera {i}: Cannot open")
                
                cap.release()
                
            except Exception as e:
                print(f"❌ Camera {i}: Error - {e}")
        
        if available_cameras:
            print(f"✅ Found {len(available_cameras)} working cameras: {available_cameras}")
            return True
        else:
            print("❌ No working cameras found")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_input_simulation():
    """Test input simulation capabilities."""
    print("\n🎮 Testing input simulation...")
    
    try:
        from pynput import keyboard
        from pynput.keyboard import Key, KeyCode
        
        # Test keyboard controller creation
        kb_controller = keyboard.Controller()
        print("✅ Keyboard controller created")
        
        # Test key object creation
        test_keys = {
            'w': KeyCode.from_char('w'),
            'space': Key.space,
            'escape': Key.esc
        }
        
        for key_name, key_obj in test_keys.items():
            print(f"✅ Key '{key_name}': {key_obj}")
        
        print("✅ Input simulation test successful")
        print("⚠️  Note: Actual key presses not tested to avoid interference")
        
        return True
        
    except Exception as e:
        print(f"❌ Input simulation test failed: {e}")
        return False

def test_configuration_system():
    """Test configuration loading and validation."""
    print("\n🔧 Testing configuration system...")
    
    try:
        # Import the configuration manager from main
        sys.path.insert(0, '/workspace')
        from main import ConfigurationManager
        
        # Test configuration creation
        config = ConfigurationManager("test_config.json")
        print("✅ Configuration manager created")
        
        # Test configuration access
        camera_width = config.get("camera.width", 1280)
        fps_target = config.get("performance.max_fps", 60)
        print(f"✅ Configuration access: camera width={camera_width}, fps={fps_target}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_system_test():
    """Run comprehensive system test suite."""
    print("🔬" * 30)
    print("MILITARY-GRADE SYSTEM COMPATIBILITY TEST")
    print("🔬" * 30)
    print("Testing all components before main system launch...")
    
    tests = [
        ("Library Imports", test_imports),
        ("MediaPipe Compatibility", test_mediapipe_compatibility),
        ("Camera Detection", test_camera_detection),
        ("Input Simulation", test_input_simulation),
        ("Configuration System", test_configuration_system)
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed_tests += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed_tests += 1
            print(f"💥 {test_name}: CRASHED - {e}")
    
    print("\n" + "🔬" * 30)
    print("TEST RESULTS SUMMARY")
    print("🔬" * 30)
    print(f"✅ Passed: {passed_tests}/{len(tests)} tests")
    print(f"❌ Failed: {failed_tests}/{len(tests)} tests")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is ready for gesture control!")
        print("🚀 You can now run: python main.py")
        return True
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed.")
        print("❗ Please resolve issues before running the main system.")
        
        if failed_tests <= 2:
            print("🔧 Minor issues detected - system may still work with reduced functionality.")
        else:
            print("🚨 Major issues detected - system may not work properly.")
        
        return False

if __name__ == "__main__":
    """Run the system test when executed directly."""
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    success = run_comprehensive_system_test()
    
    if success:
        print("\n🎮 Ready to start gesture control!")
        sys.exit(0)
    else:
        print("\n🔧 Please fix issues and run test again.")
        sys.exit(1)
