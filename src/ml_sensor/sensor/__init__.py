"""ML Sensor simulation for person detection."""

import tensorflow as tf
import numpy as np
import time
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ml_sensor.config import SensorConfig

logger = logging.getLogger(__name__)


class PersonDetectionSensor:
    """ML Sensor for Person Detection.
    
    Simulates a hardware ML sensor with I2C-like interface for person detection.
    Processes images locally and outputs semantic information (person detected)
    instead of raw image data.
    
    Features:
        - Local inference (privacy-preserving)
        - I2C-compatible interface
        - Power state management
        - Performance monitoring
        - Hardware constraint simulation
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[SensorConfig] = None,
        sensor_id: Optional[int] = None
    ):
        """Initialize ML Sensor.
        
        Args:
            model_path: Path to TFLite model file.
            config: Sensor configuration (optional).
            sensor_id: Sensor I2C address (optional, overrides config).
        """
        self.config = config or SensorConfig()
        self.sensor_id = sensor_id or self.config.sensor_id
        self.model_path = Path(model_path)
        
        # Power state
        self.power_state = self.config.default_power_state
        
        # Load TFLite model
        self._load_model()
        
        # Statistics
        self.total_inferences = 0
        self.total_inference_time_ms = 0.0
        
        logger.info(f"ML Sensor initialized (ID: 0x{self.sensor_id:02X})")
        logger.info(f"  Model: {self.model_path.name}")
        logger.info(f"  Input shape: {self.input_details[0]['shape']}")
        logger.info(f"  Power state: {self.power_state}")
    
    def _load_model(self):
        """Load TFLite model and allocate tensors."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_shape = self.input_details[0]['shape']
            self.input_dtype = self.input_details[0]['dtype']
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"  Input dtype: {self.input_dtype}")
            logger.info(f"  Output dtype: {self.output_details[0]['dtype']}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image (hardware-level preprocessing).
        
        Args:
            image: Input image (any size, RGB or grayscale).
            
        Returns:
            Preprocessed image ready for inference.
        """
        # Import here to avoid loading OpenCV unnecessarily
        import cv2
        
        # Get target size from input shape
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        
        # Resize to target size
        img = cv2.resize(image, (target_w, target_h))
        
        # Convert to grayscale if needed
        if self.input_shape[3] == 1 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        elif self.input_shape[3] == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Convert to model's expected dtype
        if self.input_dtype == np.uint8:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(self.input_dtype)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _postprocess(
        self,
        raw_output: np.ndarray,
        inference_time: float
    ) -> Dict[str, Any]:
        """Format output like a real sensor (I2C data packet).
        
        Args:
            raw_output: Raw model output.
            inference_time: Inference time in milliseconds.
            
        Returns:
            Sensor data packet (dict).
        """
        # Extract confidence value
        confidence = float(raw_output[0][0])
        
        # Scale for quantized models
        if self.output_details[0]['dtype'] == np.uint8:
            confidence = confidence / 255.0
        
        # Threshold for binary decision
        person_detected = confidence > self.config.confidence_threshold
        
        # Sensor output (mimics I2C data structure)
        sensor_data = {
            "sensor_id": f"0x{self.sensor_id:02X}",
            "person_detected": bool(person_detected),
            "confidence": float(confidence),
            "inference_time_ms": float(inference_time),
            "power_state": self.power_state,
            "timestamp": int(time.time())
        }
        
        return sensor_data
    
    def detect(
        self,
        image: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Main detection method (public sensor interface).
        
        Args:
            image: Input image (numpy array).
            verbose: Print debug information.
            
        Returns:
            Sensor data packet (dict).
        """
        if self.power_state == "sleep":
            logger.warning("Sensor is in sleep mode. Wake up sensor first.")
            return {
                "sensor_id": f"0x{self.sensor_id:02X}",
                "error": "Sensor in sleep mode",
                "power_state": self.power_state
            }
        
        # Transition to active state
        prev_state = self.power_state
        self.power_state = "active"
        
        try:
            # Preprocessing
            processed_img = self._preprocess(image)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                processed_img
            )
            
            # Inference with timing
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Post-processing
            result = self._postprocess(output, inference_time)
            
            # Update statistics
            self.total_inferences += 1
            self.total_inference_time_ms += inference_time
            
            # Return to idle state
            self.power_state = "idle"
            
            if verbose:
                print(f"\n[ML Sensor 0x{self.sensor_id:02X}] Detection Result:")
                print(json.dumps(result, indent=2))
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.power_state = prev_state
            return {
                "sensor_id": f"0x{self.sensor_id:02X}",
                "error": str(e),
                "power_state": self.power_state
            }
    
    def read_i2c(self, image: np.ndarray) -> Dict[str, Any]:
        """Alias for detect() to mimic I2C read operation.
        
        Args:
            image: Input image.
            
        Returns:
            Sensor data packet.
        """
        return self.detect(image)
    
    def set_power_state(self, state: str):
        """Set sensor power state.
        
        Args:
            state: Power state ('sleep', 'idle', 'active').
        """
        if state not in self.config.power_states:
            raise ValueError(f"Invalid power state: {state}")
        
        self.power_state = state
        logger.info(f"Power state changed to: {state}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor statistics.
        
        Returns:
            Dict with inference statistics.
        """
        avg_inference_time = (
            self.total_inference_time_ms / self.total_inferences
            if self.total_inferences > 0 else 0.0
        )
        
        return {
            "sensor_id": f"0x{self.sensor_id:02X}",
            "total_inferences": self.total_inferences,
            "average_inference_time_ms": avg_inference_time,
            "total_runtime_ms": self.total_inference_time_ms,
            "power_state": self.power_state,
            "model": str(self.model_path.name)
        }
    
    def reset(self):
        """Reset sensor statistics."""
        self.total_inferences = 0
        self.total_inference_time_ms = 0.0
        self.power_state = self.config.default_power_state
        logger.info("Sensor statistics reset")
    
    def get_datasheet(self) -> Dict[str, Any]:
        """Generate ML Sensor datasheet information.
        
        Returns:
            Datasheet information dict.
        """
        return {
            "device_info": {
                "name": "ML Sensor Person Detection",
                "sensor_id": f"0x{self.sensor_id:02X}",
                "interface": self.config.interface,
                "version": "1.0.0"
            },
            "capabilities": {
                "primary_function": "Binary person detection",
                "output_format": self.config.output_format,
                "confidence_threshold": self.config.confidence_threshold
            },
            "performance": {
                "inference_time_ms": self.config.inference_time_ms,
                "power_consumption_mw": self.config.power_consumption_mw,
                "ram_usage_kb": self.config.ram_usage_kb
            },
            "hardware": {
                "target_platform": self.config.target_platform,
                "input_shape": self.input_shape.tolist(),
                "input_dtype": str(self.input_dtype)
            }
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would require an actual model file
    # sensor = PersonDetectionSensor(model_path="models/person_detector_int8.tflite")
    
    print("ML Sensor module loaded successfully")
