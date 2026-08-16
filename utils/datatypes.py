from typing import TypeAlias, Union

import numpy as np
import torch

Device = Union[str, torch.device]
PointCloud: TypeAlias = tuple[np.ndarray, np.ndarray]
