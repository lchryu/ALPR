"""
Debug Image Logger - Instrumentation Only

This module provides optional image logging for debugging purposes.
It is PURE INSTRUMENTATION and does NOT modify pipeline logic.

When enabled=False, all operations are no-ops with zero side effects.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


class DebugImageLogger:
    """
    Optional image logger for debugging ALPR pipeline.
    
    This is instrumentation only - does not modify images, return values, or control flow.
    When enabled=False, all operations are no-ops.
    """
    
    def __init__(self, enabled: bool = False, root_dir: str = "runs/debug"):
        """
        Initialize debug logger.
        
        Args:
            enabled: If False, logger does nothing (zero side effects)
            root_dir: Root directory for debug outputs (default: runs/debug)
        """
        self.enabled = enabled
        self.root_dir = Path(root_dir)
        self.output_dir: Optional[Path] = None
        self.step_counter = 0
        
        if self.enabled:
            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            self.output_dir = self.root_dir / f"debug_{timestamp}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, image: np.ndarray):
        """
        Save image with auto-incrementing step number.
        
        When enabled=False, this is a no-op.
        
        Args:
            name: Descriptive name for the image (e.g., "deskew", "pass1_input")
            image: Image array to save (BGR, grayscale, or binary)
        """
        if not self.enabled:
            return  # No-op when disabled
        
        if self.output_dir is None:
            print(f"Warning: Debug logger output_dir is None, cannot save {name}")
            return  # Safety check
        
        try:
            # Auto-number with step counter
            filename = f"{self.step_counter:03d}_{name}.jpg"
            filepath = self.output_dir / filename
            
            # Save image (handles BGR, grayscale, binary)
            success = cv2.imwrite(str(filepath), image)
            if not success:
                print(f"Warning: Failed to save debug image: {filepath}")
            else:
                print(f"Debug: Saved {name} to {filepath}")
            
            self.step_counter += 1
        except Exception as e:
            print(f"Error saving debug image {name}: {e}")

