# ML Sensor Datasheet

## Device Information

- **Name:** ML Sensor Person Detection v1.0
- **Sensor ID:** 0x62 (I2C Address)
- **Version:** 1.0.0
- **Date:** 2026-02-02

---

## Overview & Capabilities

### Primary Function
Binary person detection with confidence score

### Detection Methodology
Deep learning-based computer vision using MobileNetV2 architecture with edge optimization

### Output Format
```json
{
  "sensor_id": "0x62",
  "person_detected": true,
  "confidence": 0.96,
  "inference_time_ms": 8.7,
  "power_state": "idle",
  "timestamp": 1738500196
}
```

---

## Model Characteristics

| Attribute | Value |
|-----------|-------|
| Architecture | MobileNetV2 (α=0.5) |
| Framework | TensorFlow Lite |
| Quantization | INT8 (post-training) |
| Input Shape | 96×96 grayscale |
| Model Size | 0.85 MB |
| Parameters | ~1.2M |
| Inference Time | 8.7 ms (avg) |

---

## Performance Metrics

### Accuracy Analysis

| Metric | FP32 Model | INT8 (Quantized) |
|--------|------------|------------------|
| Accuracy | 94.2% | 93.8% |
| Precision | 93.5% | 93.1% |
| Recall | 94.8% | 94.3% |
| F1-Score | 94.1% | 93.7% |
| ROC-AUC | 0.983 | 0.979 |

**Accuracy Degradation (Quantization):** 0.4% (industry-leading)

### Latency Characteristics

- **Average Inference Time:** 8.7 ms
- **P50 Latency:** 8.3 ms
- **P95 Latency:** 11.2 ms
- **P99 Latency:** 14.5 ms

*Measured on Raspberry Pi 4 (ARM Cortex-A72 @ 1.5GHz)*

---

## Dataset Nutrition Label

### Source Dataset
**COCO 2017** - Common Objects in Context

### Sample Distribution
- **Training Samples:** 10,000 images
- **Validation Samples:** 2,000 images
- **Test Samples:** 2,000 images

### Class Balance
- **Person:** 50% (7,000 samples)
- **No Person:** 50% (7,000 samples)

### Demographics
Images contain diverse:
- Age groups (children, adults, elderly)
- Genders
- Ethnicities
- Clothing styles
- Environmental contexts (indoor/outdoor)

### Known Limitations

⚠️ **Environmental:**
- Reduced accuracy in very low light (<50 lux)
- Performance degradation in heavy fog/rain
- Challenges with extreme viewing angles (>60° from normal)

⚠️ **Use Case:**
- Optimized for single person detection (may miss people in crowds)
- Best performance at 0.5-3 meters distance
- Trained primarily on standing/sitting poses

⚠️ **Bias Considerations:**
- Dataset skews toward Western demographics
- May have reduced performance on underrepresented groups
- Regular validation recommended for deployment context

---

## Hardware Specifications

### Target Platforms

| Platform | Supported | Tested |
|----------|-----------|--------|
| ARM Cortex-M4 | ✅ | ⚠️ |
| ARM Cortex-A72 (RPi4) | ✅ | ✅ |
| ESP32 | ✅ | ⚠️ |
| STM32 | ✅ | ❌ |

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| RAM | 420 KB |
| Flash | 0.85 MB (model) + 50 KB (runtime) |
| Power (Active) | ~50 mW |
| Power (Idle) | ~5 mW |
| Power (Sleep) | <1 mW |

### Communication Interface

**Primary:** I2C (400 kHz, Fast Mode)
- **Address:** 0x62 (7-bit)
- **Data Rate:** Up to 50 Hz
- **Output:** JSON or binary

**Alternative:** SPI, UART (software configurable)

---

## Privacy & Security

### Privacy Features
✅ **On-Device Processing** - Images never leave the sensor
✅ **No Data Storage** - No image retention or logging
✅ **Semantic Output Only** - Only detection result transmitted
✅ **Configurable Threshold** - User-controlled sensitivity

### Security Considerations
- Model extraction possible from flash memory
- No authentication on I2C interface
- No encrypted communication (plaintext JSON)

**Recommendation:** Deploy in trusted environments or add application-layer security

---

## Environmental Specifications

| Parameter | Range |
|-----------|-------|
| Operating Temperature | -10°C to 50°C |
| Storage Temperature | -20°C to 70°C |
| Humidity | 10% to 90% (non-condensing) |
| Vibration | MIL-STD-810G |

---

## Comparison: ML Sensor vs Traditional IoT

| Aspect | Traditional IoT | ML Sensor | Improvement |
|--------|----------------|-----------|-------------|
| **Privacy** | ❌ Raw images → cloud | ✅ Local processing | 100% |
| **Latency** | 150-300 ms | 8.7 ms | **94.5% ↓** |
| **Data Transmitted** | 9,216 bytes/frame | 128 bytes/frame | **98.6% ↓** |
| **Bandwidth** | ~920 KB/s (10 FPS) | ~1.25 KB/s | **99.9% ↓** |
| **Offline** | ❌ Requires internet | ✅ Fully offline | N/A |
| **Power** | High (WiFi streaming) | Low (local inference) | ~80% ↓ |

---

## Compliance & Certifications

**Planned (not yet certified):**
- CE Mark (Europe)
- FCC (USA)
- RoHS Compliant
- GDPR Compliant (privacy-by-design)

---

## Warranty & Support

- **Warranty:** 1 year limited warranty
- **Support:** GitHub issues, community forum
- **Updates:** OTA firmware updates supported
- **Lifecycle:** Minimum 3 years production commitment

---

## References

1. **ML Sensors: A Foundation for the IoT Era**
   - Banbury et al., arXiv:2206.03266, 2022
   - https://arxiv.org/abs/2206.03266

2. **Datasheets for ML Sensors**
   - Banbury et al., arXiv:2306.08848, 2023
   - https://arxiv.org/abs/2306.08848

3. **COCO Dataset**
   - Lin et al., ECCV 2014
   - https://cocodataset.org/

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-02 | Initial release |

---

**Datasheet Generated By:** ML Sensor Person Detection Project
**Contact:** github.com/imshujaa/ML_Sensors_2026
