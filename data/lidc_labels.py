"""LIDC-IDRI label derivation (pure Python, no pylidc needed so it can be unit-tested)."""
from typing import Dict, List, Optional
import numpy as np


def consensus_label(
    ratings: List[int],
    min_annotators: int = 3,
) -> Optional[Dict[str, float]]:
    """
    Radiologist malignancy ratings (1-5, 'highly unlikely' .. 'highly suspicious') -> binary label.

    median < 3 -> benign (0); median > 3 -> malignant (1); median == 3 -> excluded (None),
    as is any nodule rated by fewer than min_annotators readers.

    CAVEAT: this is a *perceived-malignancy* label, not pathology. Only a minority of LIDC
    nodules have a diagnosis. Say so wherever results are reported.
    """
    r = [int(x) for x in ratings if x is not None and 1 <= int(x) <= 5]
    if len(r) < min_annotators:
        return None
    med = float(np.median(r))
    if med == 3.0:
        return None
    return {"label": int(med > 3.0), "median_malignancy": med,
            "mean_malignancy": float(np.mean(r)), "n_annotators": len(r)}
