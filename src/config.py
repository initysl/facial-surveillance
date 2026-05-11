import yaml
import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration loader with environment variable support."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve environment variables
        config = Config._resolve_env_vars(config)
        
        # Validate required fields
        Config._validate(config)
        
        return config
    
    @staticmethod
    def _resolve_env_vars(config: Dict) -> Dict:
        """Replace ${VAR} with environment variable values."""
        
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
        
        return resolve_value(config)  # type: ignore
    
    @staticmethod
    def _validate(config: Dict):
        """Validate configuration."""
        
        # Required top-level keys
        required = ['model', 'matching', 'processing', 'storage']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
        
        # Validate device
        if config['model']['device'] not in ['cuda', 'cpu']:
            raise ValueError("model.device must be 'cuda' or 'cpu'")
        
        # Validate threshold
        threshold = config['matching']['threshold']
        if not 0 <= threshold <= 1:
            raise ValueError("matching.threshold must be between 0 and 1")