import platform
import psutil
import os


def print_env_details():
    """
    Print system details (OS, RAM, CUDA, packages, etc.) to stdout.
    """
    import torch
    import pytorch_lightning as pl
    import numpy as np
    import pandas as pd
    import sklearn
    import transformers

    def info_system():
        return {
            "OS": platform.system(),
            "architecture": platform.architecture(),
            "version": platform.version(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "ram": psutil.virtual_memory().total,
        }

    def info_cuda():
        return {
            "GPU": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
        }

    def info_packages():
        return {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyTorch_version": torch.__version__,
            "pyTorch_debug": torch.version.debug,
            "pytorch-lightning": pl.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
        }

    def nice_print(details, level=0):
        LEVEL_OFFSET = "\t"
        KEY_PADDING = 20
        lines = []
        for k in sorted(details):
            key = f"* {k}:" if level == 0 else f"- {k}:"
            if isinstance(details[k], dict):
                lines += [level * LEVEL_OFFSET + key]
                lines += nice_print(details[k], level + 1)
            elif isinstance(details[k], (set, list, tuple)):
                lines += [level * LEVEL_OFFSET + key]
                lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
            else:
                template = "{:%is} {}" % KEY_PADDING
                key_val = template.format(key, details[k])
                lines += [(level * LEVEL_OFFSET) + key_val]
        return lines

    details = {
        "System": info_system(),
        "CUDA": info_cuda(),
        "Packages": info_packages(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(text)

