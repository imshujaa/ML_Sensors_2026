#!/usr/bin/env python3
"""Evaluation script for ML Sensor Person Detection."""

import argparse
import logging
import sys
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_sensor.config import Config, get_default_config
from ml_sensor.data import PersonDetectionDataset
from ml_sensor.evaluation import MetricsCalculator
from ml_sensor.sensor import PersonDetectionSensor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_keras_model(model_path: str, config: Config):
    """Evaluate Keras model.
    
    Args:
        model_path: Path to Keras model.
        config: Configuration object.
    """
    logger.info("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    # Get test dataset
    dataset_handler = PersonDetectionDataset(config.data)
    _, _, test_ds = dataset_handler.get_datasets()
    
    # Evaluate
    logger.info("Evaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    
    # Get predictions for detailed metrics
    y_true = []
    y_pred_prob = []
    
    for batch_x, batch_y in test_ds:
        y_true.append(batch_y.numpy())
        y_pred_prob.append(model.predict(batch_x, verbose=0))
    
    y_true = np.concatenate(y_true)
    y_pred_prob = np.concatenate(y_pred_prob)
    
    # Calculate all metrics
    metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred_prob)
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    for metric, value in metrics.items():
        logger.info(f"{metric:.<30} {value}")
    logger.info("=" * 70)
    
    return metrics, y_true, y_pred_prob


def evaluate_tflite_model(model_path: str, config: Config):
    """Evaluate TFLite model.
    
    Args:
        model_path: Path to TFLite model.
        config: Configuration object.
    """
    logger.info(f"Initializing ML Sensor with model: {model_path}")
    sensor = PersonDetectionSensor(model_path=model_path, config=config.sensor)
    
    # Get test dataset
    dataset_handler = PersonDetectionDataset(config.data)
    _, _, test_ds = dataset_handler.get_datasets()
    
    # Evaluate
    logger.info("Evaluating ML Sensor...")
    y_true = []
    y_pred_prob = []
    inference_times = []
    
    for batch_x, batch_y in test_ds:
        for i in range(len(batch_y)):
            img = (batch_x[i].numpy() * 255).astype(np.uint8)
            result = sensor.detect(img)
            
            if 'error' not in result:
                y_true.append(batch_y[i].numpy())
                y_pred_prob.append(result['confidence'])
                inference_times.append(result['inference_time_ms'])
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred_prob)
    metrics['avg_inference_time_ms'] = float(np.mean(inference_times))
    metrics['model_size_mb'] = Path(model_path).stat().st_size / (1024 * 1024)
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("ML SENSOR EVALUATION RESULTS")
    logger.info("=" * 70)
    for metric, value in metrics.items():
        logger.info(f"{metric:.<30} {value}")
    logger.info("=" * 70)
    
    # Print sensor statistics
    stats = sensor.get_statistics()
    logger.info("\nSensor Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return metrics, y_true, y_pred_prob


def generate_visualizations(y_true, y_pred_prob, output_dir: str = "results"):
    """Generate evaluation visualizations.
    
    Args:
        y_true: True labels.
        y_pred_prob: Predicted probabilities.
        output_dir: Output directory for visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_true.flatten()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['No Person', 'Person'])
    plt.yticks([0.5, 1.5], ['No Person', 'Person'])
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    logger.info(f"Visualizations saved to: {output_dir}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ML Sensor Person Detection Model")
    parser.add_argument('--model-path', type=str, required=True, help='Path to model file (.h5 or .tflite)')
    parser.add_argument('--generate-report', action='store_true', help='Generate evaluation report')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    
    # Evaluate based on model type
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    if model_path.suffix == '.tflite':
        metrics, y_true, y_pred_prob = evaluate_tflite_model(str(model_path), config)
    else:
        metrics, y_true, y_pred_prob = evaluate_keras_model(str(model_path), config)
    
    # Generate visualizations
    if args.generate_report:
        generate_visualizations(y_true, y_pred_prob, args.output_dir)
        
        # Save metrics to JSON
        metrics_path = Path(args.output_dir) / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
