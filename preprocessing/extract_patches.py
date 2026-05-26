import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

def main():
    cfg = load_config()
    input_path = cfg["data"]["image_path"]
    output_dir = Path(cfg["data"]["patches_dir"])
    patch_size = cfg["training"]["patch_size"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(input_path).exists():
        logger.error("%s not found. Run download_sentinel.py first.", input_path)
        return

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        logger.info("Processing image: %dx%d pixels", width, height)

        patch_id = 0
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                window = Window(x, y, patch_size, patch_size)
                patch = src.read(window=window)

                if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                    if np.count_nonzero(patch) > (patch_size * patch_size * 0.5):
                        patch_file = output_dir / f"patch_{patch_id:05d}.npy"
                        np.save(patch_file, patch)
                        patch_id += 1

    logger.info("Saved %d patches to %s", patch_id, output_dir)

if __name__ == "__main__":
    main()
