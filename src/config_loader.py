import importlib.util
import sys
import os

# This module provides a robust way to load the framework's configuration.
# It ensures that the analysis scripts can be run both by the main
# conductor script and as standalone files for development and debugging.

def load_config():
    """
    Dynamically loads the correct configuration module.

    The priority is as follows:
    1. If `src/config.py` exists, load it. (This is the file created by the conductor).
    2. If not, fall back to loading `src/config_full.py` (for standalone runs).
    3. If neither can be found, raise a clear error.

    Returns:
        The loaded configuration module object.
    """
    config_path = os.path.join('src', 'config.py')
    config_full_path = os.path.join('src', 'config_full.py')

    if os.path.exists(config_path):
        target_path = config_path
        module_name = 'src.config'
    elif os.path.exists(config_full_path):
        target_path = config_full_path
        module_name = 'src.config_full'
    else:
        raise FileNotFoundError(
            "❌ CRITICAL ERROR: Could not find a configuration file.\n"
            "Please ensure that 'src/config_full.py' exists, or run the main 'run_full_analysis.py' script."
        )

    # --- Standard Python boilerplate for dynamically importing a module ---
    spec = importlib.util.spec_from_file_location(module_name, target_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)
    # --- End of boilerplate ---

    return config_module