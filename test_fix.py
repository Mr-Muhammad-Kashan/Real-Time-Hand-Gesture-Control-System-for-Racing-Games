#!/usr/bin/env python3
"""
Quick test script to verify the MediaPipe compatibility fix
"""

import sys
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    
    print("Testing MediaPipe compatibility...")
    
    # Test MediaPipe Hands initialization
    mp_hands = mp.solutions.hands
    
    # Test with configuration that should work with newer versions
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
        print("✓ MediaPipe Hands initialized with model_complexity")
    except TypeError:
        # Remove model_complexity if not supported
        hands_config.pop('model_complexity', None)
        hands = mp_hands.Hands(**hands_config)
        print("✓ MediaPipe Hands initialized (model_complexity not supported in this version)")
    
    # Test basic processing
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = hands.process(test_frame)
    print("✓ MediaPipe processing test successful")
    
    hands.close()
    print("\n✅ All tests passed! The system should work now.")
    print("Run: python main.py")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
