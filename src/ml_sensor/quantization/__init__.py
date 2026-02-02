"""Quantization utilities for ML models."""

import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from ml_sensor.config import QuantizationConfig

logger = logging.getLogger(__name__)


def quantize_model(
    model: tf.keras.Model,
    config: QuantizationConfig,
    representative_data: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Quantize Keras model to TFLite INT8.
    
    Args:
        model: Keras model to quantize.
        config: Quantization configuration.
        representative_data: Representative dataset for calibration.
        output_path: Path to save quantized model (optional).
        
    Returns:
        Quantized TFLite model (bytes).
    """
    logger.info(f"Starting quantization with method: {config.method}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if config.method == "ptq" and representative_data is not None:
        # Post-Training Quantization with representative dataset
        def representative_dataset():
            for i in range(min(config.calibration_samples, len(representative_data))):
                yield [representative_data[i:i+1].astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        logger.info(f"Using {config.calibration_samples} samples for calibration")
    
    # Convert
    tflite_model = converter.convert()
    
    logger.info(f"Quantization complete. Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"Saved quantized model to: {output_path}")
    
    return tflite_model
