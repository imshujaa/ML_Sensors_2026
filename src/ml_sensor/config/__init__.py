"""Configuration management for ML Sensor."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    
    dataset_name: str = "coco/2017"
    img_size: int = 96
    batch_size: int = 32
    train_samples: int = 10000
    val_samples: int = 2000
    test_samples: int = 2000
    use_grayscale: bool = True
    num_classes: int = 2
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle_buffer: int = 1000
    prefetch_buffer: int = 4
    cache: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    aug_rotation_range: float = 15.0
    aug_brightness_delta: float = 0.2
    aug_contrast_range: Tuple[float, float] = (0.8, 1.2)
    aug_flip_horizontal: bool = True
    aug_flip_vertical: bool = False
    aug_cutout_size: int = 16
    aug_mixup_alpha: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    architecture: str = "mobilenetv2"  # mobilenetv2, mobilenetv3, efficientnet, custom
    input_shape: Tuple[int, int, int] = (96, 96, 1)
    alpha: float = 0.5  # Width multiplier for MobileNet
    dropout_rate: float = 0.3
    use_transfer_learning: bool = False
    weights: Optional[str] = None  # "imagenet" or None
    freeze_base: bool = False
    activation: str = "relu"
    
    # Custom architecture settings
    custom_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    custom_kernel_size: int = 3


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    epochs: int = 50
    initial_lr: float = 0.001
    optimizer: str = "adam"  # adam, sgd, rmsprop
    loss: str = "binary_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])
    
    # Learning rate schedule
    lr_schedule: str = "cosine"  # cosine, onecycle, reduce_on_plateau, constant
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Regularization
    l2_regularization: float = 1e-4
    label_smoothing: float = 0.1
    
    # Advanced training
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    save_best_only: bool = True
    save_weights_only: bool = False
    monitor_metric: str = "val_accuracy"
    monitor_mode: str = "max"


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    
    method: str = "ptq"  # ptq (post-training), qat (quantization-aware training)
    target_dtype: str = "int8"  # int8, float16
    calibration_samples: int = 1000
    
    # QAT specific
    qat_epochs: int = 10
    qat_lr: float = 1e-5
    
    # Optimization
    optimize_ops: List[str] = field(default_factory=lambda: ["TFLITE_BUILTINS_INT8"])
    inference_input_type: str = "uint8"
    inference_output_type: str = "uint8"
    
    # Performance targets
    max_accuracy_degradation: float = 0.02  # 2%
    target_model_size_mb: float = 1.0
    target_inference_time_ms: float = 10.0


@dataclass
class SensorConfig:
    """ML Sensor hardware simulation configuration."""
    
    sensor_id: int = 0x62
    interface: str = "i2c"  # i2c, spi, uart
    i2c_address: int = 0x62
    
    # Power states
    power_states: List[str] = field(default_factory=lambda: ["sleep", "idle", "active"])
    default_power_state: str = "idle"
    
    # Performance characteristics
    inference_time_ms: float = 8.7
    power_consumption_mw: float = 50.0
    ram_usage_kb: int = 420
    
    # Hardware platform
    target_platform: str = "arm_cortex_m4"  # arm_cortex_m4, esp32, rpi4
    
    # Communication
    output_format: str = "json"  # json, binary, protobuf
    confidence_threshold: float = 0.5
    enable_interrupts: bool = True


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    # Project metadata
    project_name: str = "ML Sensor Person Detection"
    version: str = "1.0.0"
    seed: int = 42
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    # Experiment tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.model_dir, self.results_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Config: Default configuration object.
    """
    return Config()


def get_config_from_yaml(yaml_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file.
        
    Returns:
        Config: Configuration object loaded from YAML.
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config with nested dataclasses
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    quantization_config = QuantizationConfig(**config_dict.get('quantization', {}))
    sensor_config = SensorConfig(**config_dict.get('sensor', {}))
    
    main_config_dict = {k: v for k, v in config_dict.items() 
                        if k not in ['data', 'model', 'training', 'quantization', 'sensor']}
    
    return Config(
        **main_config_dict,
        data=data_config,
        model=model_config,
        training=training_config,
        quantization=quantization_config,
        sensor=sensor_config
    )
