"""RAA-F Route B: LiberoInputs with the target crop in the third image slot (right_wrist_0_rgb), mask True.
openpi's LiberoInputs (libero_policy.py:62, 68) fills that slot with zeros and mask False for pi0.5."""
import dataclasses

import numpy as np
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class LiberoInputsB(libero_policy.LiberoInputs):
    def __call__(self, data: dict) -> dict:
        out = super().__call__(data)
        crop = libero_policy._parse_image(data["observation/crop_image"])        # KeyError if absent: never a silent masked slot
        out["image"]["right_wrist_0_rgb"] = crop
        out["image_mask"]["right_wrist_0_rgb"] = np.True_
        return out
