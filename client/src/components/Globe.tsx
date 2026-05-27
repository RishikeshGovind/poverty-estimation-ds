import { useEffect, useRef } from "react";
import { Viewer, Entity, PointGraphics, ImageryLayer } from "resium";
import * as Cesium from "cesium";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

// Poverty rate → choropleth colour (red = high poverty, yellow-green = low)
function povertyColor(rate: number | null, opacity: number): Cesium.Color {
  const r = rate ?? 50;
  const t = Math.min(r / 80, 1); // 0 (low) → 1 (high poverty)
  return new Cesium.Color(
    0.6 + 0.4 * t,           // R: 0.6–1.0
    0.6 * (1 - t),           // G: 0.6–0
    0.1,
    opacity
  );
}

function conflictColor(fatalities: number): Cesium.Color {
  if (fatalities > 15) return Cesium.Color.RED.withAlpha(1);
  if (fatalities > 5)  return Cesium.Color.ORANGE.withAlpha(0.9);
  return Cesium.Color.YELLOW.withAlpha(0.85);
}

interface Props {
  onCountryClick: (f: PovertyFeature) => void;
}

export default function Globe({ onCountryClick }: Props) {
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const {
    layers, povertyFeatures, conflictEvents, flyTo, setFlyTo,
  } = useGlobeStore();

  // Fly-to effect
  useEffect(() => {
    if (!flyTo || !viewerRef.current) return;
    const [lat, lon, height] = flyTo;
    viewerRef.current.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
      duration: 2.0,
    });
    setFlyTo(null);
  }, [flyTo, setFlyTo]);

  // NASA GIBS MODIS NTL tile provider (2016-onward monthly composites)
  const nightlightsProvider = new Cesium.WebMapTileServiceImageryProvider({
    url: "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/VIIRS_Black_Marble_Annual_2023/default/2023-01-01/500m/{TileMatrixSet}/{TileRow}/{TileCol}.jpg",
    layer: "VIIRS_Black_Marble_Annual_2023",
    style: "default",
    format: "image/jpeg",
    tileMatrixSetID: "500m",
    maximumLevel: 8,
    tilingScheme: new Cesium.GeographicTilingScheme(),
    credit: "NASA GSFC / GIBS",
  });

  // NASA GIBS NDVI MODIS
  const ndviProvider = new Cesium.WebMapTileServiceImageryProvider({
    url: "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_NDVI_8Day/default/2023-01-01/250m/{TileMatrixSet}/{TileRow}/{TileCol}.png",
    layer: "MODIS_Terra_NDVI_8Day",
    style: "default",
    format: "image/png",
    tileMatrixSetID: "250m",
    maximumLevel: 8,
    tilingScheme: new Cesium.GeographicTilingScheme(),
    credit: "NASA GSFC / GIBS",
  });

  return (
    <Viewer
      full
      ref={(v) => {
        if (v?.cesiumElement) viewerRef.current = v.cesiumElement;
      }}
      timeline={false}
      animation={false}
      baseLayerPicker={false}
      geocoder={false}
      homeButton={false}
      sceneModePicker={false}
      navigationHelpButton={false}
      infoBox={false}
      selectionIndicator={false}
      creditContainer={document.createElement("div")} // hide credit overlay
      terrainProvider={new Cesium.EllipsoidTerrainProvider()}
      scene3DOnly
      requestRenderMode={false}
    >
      {/* Dark base map */}
      <ImageryLayer
        imageryProvider={
          new Cesium.UrlTemplateImageryProvider({
            url: "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
            credit: "CartoDB",
          })
        }
      />

      {/* Nighttime lights layer */}
      {layers.nightlights.enabled && (
        <ImageryLayer
          imageryProvider={nightlightsProvider}
          alpha={layers.nightlights.opacity}
        />
      )}

      {/* NDVI layer */}
      {layers.ndvi.enabled && (
        <ImageryLayer
          imageryProvider={ndviProvider}
          alpha={layers.ndvi.opacity}
        />
      )}

      {/* Poverty choropleth dots */}
      {layers.poverty.enabled &&
        povertyFeatures.map((f) => (
          <Entity
            key={f.iso3}
            position={Cesium.Cartesian3.fromDegrees(f.lon, f.lat)}
            onClick={() => onCountryClick(f)}
          >
            <PointGraphics
              pixelSize={18}
              color={povertyColor(f.poverty_rate, layers.poverty.opacity)}
              outlineColor={Cesium.Color.WHITE.withAlpha(0.3)}
              outlineWidth={1}
              scaleByDistance={new Cesium.NearFarScalar(1.5e6, 1.5, 8e6, 0.5)}
            />
          </Entity>
        ))}

      {/* Conflict events */}
      {layers.conflict.enabled &&
        conflictEvents.map((ev) => (
          <Entity
            key={ev.id}
            position={Cesium.Cartesian3.fromDegrees(ev.lon, ev.lat)}
            description={`<b>${ev.event_type}</b><br/>${ev.notes}<br/>Fatalities: ${ev.fatalities}<br/>${ev.date}`}
          >
            <PointGraphics
              pixelSize={10}
              color={conflictColor(ev.fatalities).withAlpha(layers.conflict.opacity)}
              outlineColor={Cesium.Color.RED.withAlpha(0.6)}
              outlineWidth={2}
              scaleByDistance={new Cesium.NearFarScalar(1e6, 1.8, 8e6, 0.6)}
            />
          </Entity>
        ))}
    </Viewer>
  );
}
