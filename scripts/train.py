#!/usr/bin/env python3
"""Training script for ML Sensor Person Detection."""

import argparse
import logging
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_sensor.config import Config, get_config_from_yaml, get_default_config
from ml_sensor.data import PersonDetectionDataset
from ml_sensor.models import create_model
from ml_sensor.quantization import quantize_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(config: Config):
    """Train the person detection model.
    
    Args:
        config: Configuration object.
    """
    logger.info("=" * 70)
    logger.info("ML SENSOR PERSON DETECTION - TRAINING")
    logger.info("=" * 70)
    
    # Set random seeds
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create dataset
    logger.info("\n[1/6] Loading datasets...")
    dataset_handler = PersonDetectionDataset(config.data)
    train_ds, val_ds, test_ds = dataset_handler.get_datasets()
    
    # Create model
    logger.info("\n[2/6] Building model...")
    model = create_model(config.model)
    
    # Compile model
    logger.info("\n[3/6] Compiling model...")
    metrics = [
        config.training.metrics[0],  # accuracy
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.training.initial_lr),
        loss=config.training.loss,
        metrics=metrics
    )
    
    model.summary()
    
    # Setup callbacks
    logger.info("\n[4/6] Setting up callbacks...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(config.model_dir) / 'best_model.h5'),
            save_best_only=config.training.save_best_only,
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.training.reduce_lr_factor,
            patience=config.training.reduce_lr_patience,
            min_lr=config.training.lr_min,
            verbose=1
        )
    ]
    
    if config.use_tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=config.logs_dir,
                histogram_freq=1
            )
        )
    
    # Train model
    logger.info("\n[5/6] Training model...")
    logger.info(f"Training for {config.training.epochs} epochs")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    logger.info("\n[6/6] Evaluating on test set...")
    test_results = model.evaluate(test_ds, verbose=1)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Test Loss: {test_results[0]:.4f}")
    logger.info(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    logger.info(f"Test Precision: {test_results[2]:.4f}")
    logger.info(f"Test Recall: {test_results[3]:.4f}")
    logger.info(f"F1-Score: {2 * test_results[2] * test_results[3] / (test_results[2] + test_results[3]):.4f}")
    
    # Save final model
    final_model_path = Path(config.model_dir) / 'final_model.h5'
    model.save(final_model_path)
    logger.info(f"\nFinal model saved to: {final_model_path}")
    
    # Quantize model
    logger.info("\nQuantizing model to INT8...")
    
    # Get representative data for calibration
    calibration_data = []
    for batch in train_ds.take(config.quantization.calibration_samples // config.data.batch_size + 1):
        calibration_data.append(batch[0].numpy())
    calibration_data = np.concatenate(calibration_data, axis=0)[:config.quantization.calibration_samples]
    
    quantized_model_path = Path(config.model_dir) / 'person_detector_int8.tflite'
    quantize_model(
        model=model,
        config=config.quantization,
        representative_data=calibration_data,
        output_path=str(quantized_model_path)
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL DONE! 🎉")
    logger.info("=" * 70)
    logger.info(f"✓ FP32 Model: {final_model_path}")
    logger.info(f"✓ INT8 Model: {quantized_model_path}")
    logger.info(f"✓ Logs: {config.logs_dir}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ML Sensor Person Detection Model")
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--architecture', type=str, help='Model architecture (mobilenetv2, mobilenetv3, custom)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Initial learning rate')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = get_config_from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.architecture:
        config.model.architecture = args.architecture
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.initial_lr = args.learning_rate
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
