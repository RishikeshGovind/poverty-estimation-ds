# visualization/generate_heatmap.py
# Generate interactive heatmap from model predictions
# Uses folium, pandas, geopandas

import pandas as pd
import folium
from folium.plugins import HeatMap

# 1. Load prediction CSV (should have latitude, longitude, prediction columns)
pred_csv = "data/predictions.csv"  # Update path if needed
df = pd.read_csv(pred_csv)

# 2. Create base map centered on mean location
center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=6)

# 3. Generate heatmap data (latitude, longitude, prediction)
heat_data = df[['latitude', 'longitude', 'prediction']].values.tolist()
HeatMap(heat_data, radius=12, blur=15, max_zoom=1).add_to(m)

# 4. Add markers for each point
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=f"Prediction: {row['prediction']:.2f}"
    ).add_to(m)

# 5. Save interactive map as HTML
output_map = "prediction_heatmap.html"
m.save(output_map)
print(f"Interactive heatmap saved as {output_map}")
