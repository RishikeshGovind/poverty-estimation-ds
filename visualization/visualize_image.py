# visualization/visualize_image.py
# Simple script to load and display a Sentinel-2 RGB GeoTIFF
# Uses rasterio and matplotlib

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Open the GeoTIFF file
    geotiff_path = "data/raw/kenya_sentinel2.tif" 
    with rasterio.open(geotiff_path) as src:
        # 2. Read RGB bands (assume bands 1=Red, 2=Green, 3=Blue)
        rgb = src.read([1, 2, 3])  # Shape: (3, height, width)
        print("Image dimensions: {} x {}".format(src.width, src.height))
        print("Number of bands:", src.count)
        print("Band descriptions:", src.descriptions)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

    # 3. Normalize values to 0-1 for display
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)

    # 4. Rearrange to (height, width, 3) for matplotlib
    rgb_img = np.transpose(rgb_norm, (1, 2, 0))

    # 5. Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_img)
    plt.title("Sentinel-2 RGB Image")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
