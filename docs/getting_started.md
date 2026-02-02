# Getting Started with ML Sensor Person Detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- (Optional) CUDA-capable GPU for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/imshujaa/ML_Sensors_2026.git
cd ML_Sensors_2026
```

### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Package in Development Mode

```bash
pip install -e .
```

## Quick Start

### Generate Sample Data

Before training, generate sample data from the COCO dataset:

```bash
python scripts/generate_sample_data.py
```

This will download and prepare a small sample dataset (~200 images) for testing.

### Train a Model

Train with default configuration:

```bash
python scripts/train.py
```

Train with custom parameters:

```bash
python scripts/train.py --epochs 30 --batch-size 64 --architecture mobilenetv3
```

### Evaluate the Model

Evaluate a trained model:

```bash
python scripts/evaluate.py --model-path models/best_model.h5 --generate-report
```

Evaluate the quantized TFLite model:

```bash
python scripts/evaluate.py --model-path models/person_detector_int8.tflite --generate-report
```

### Use the ML Sensor

```python
from ml_sensor.sensor import PersonDetectionSensor
import cv2

# Initialize sensor
sensor = PersonDetectionSensor(
    model_path="models/person_detector_int8.tflite"
)

# Load test image
image = cv2.imread("test_image.jpg")

# Detect person
result = sensor.detect(image, verbose=True)

print(f"Person detected: {result['person_detected']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Inference time: {result['inference_time_ms']:.1f} ms")
```

## Configuration

Customize training by editing `configs/default_config.yaml` or creating your own YAML file:

```yaml
model:
  architecture: mobilenetv2  # or mobilenetv3, efficientnet, custom
  alpha: 0.5
  dropout_rate: 0.3

training:
  epochs: 50
  initial_lr: 0.001
  batch_size: 32
```

Then train with custom config:

```bash
python scripts/train.py --config configs/my_config.yaml
```

## Next Steps

- **Training**: See [Training Guide](training_guide.md) for detailed training options
- **Deployment**: See [Deployment Guide](deployment_guide.md) for edge deployment
- **API Reference**: See [API Reference](api_reference.md) for detailed API documentation
- **Benchmarks**: See [Benchmarks](benchmarks.md) for performance comparisons

## Troubleshooting

### CUDA/GPU Issues

If you encounter GPU-related errors, install the CPU-only version of TensorFlow:

```bash
pip install tensorflow-cpu
```

### Dataset Download Issues

If COCO dataset download is slow or fails:
1. Try a different network connection
2. Use a VPN if geo-restricted
3. Manually download from [COCO website](https://cocodataset.org/)

### Memory Issues

If you run out of memory during training:
- Reduce `batch_size` in config
- Reduce `train_samples` for faster iterations
- Use a smaller image size (e.g., `img_size: 64`)

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/imshujaa/ML_Sensors_2026/issues)
- Check existing documentation in the `docs/` folder
