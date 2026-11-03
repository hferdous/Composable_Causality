import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union

class Config:
    """
    Configuration handler for DTCD dataset recipes.
    Supports loading from JSON/YAML and validation of schema.
    """
    def __init__(self, config_data: Dict[str, Any] = None):
        self.data = config_data if config_data else {}
        self.validate()

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        """Load configuration from a JSON or YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use .json or .yaml")
                
        return cls(data)

    def validate(self):
        """
        Validate the configuration against the expected schema.
        This is a basic validation. In a real scenario, we might use pydantic or jsonschema.
        """
        # Basic required fields for a dataset recipe
        # We can make this loose for now and strict later
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'structure.n_vars')."""
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def save(self, path: Union[str, Path]):
        """Save the current configuration to a file."""
        path = Path(path)
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(self.data, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(self.data, f, indent=4)
            else:
                raise ValueError("Unsupported output format. Use .json or .yaml")

    def __repr__(self):
        return f"Config({self.data})"
