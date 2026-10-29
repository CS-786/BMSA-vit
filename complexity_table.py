"""Parameter and FLOP table for BMSA-ViT settings (Keras count + closed form).

    python scripts/complexity_table.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmsa_vit.model import build_bmsa_vit, count_macs, count_params  # noqa: E402

SETTINGS = [(48, 16), (48, 8), (48, 4), (48, 2), (224, 16), (224, 8), (224, 4), (224, 2)]


def main() -> None:
    print(f"{'image/patch':<12}{'tokens':>8}{'params (Keras)':>16}{'params (formula)':>18}{'GFLOPs (2xMAC)':>16}")
    for s, p in SETTINGS:
        m = build_bmsa_vit(image_size=s, patch_size=p)
        print(f"{f'{s}/{p}':<12}{(s // p) ** 2:>8}{m.count_params():>16,}{count_params(s, p):>18,}"
              f"{2 * count_macs(s, p) / 1e9:>16.4f}")


if __name__ == "__main__":
    main()
