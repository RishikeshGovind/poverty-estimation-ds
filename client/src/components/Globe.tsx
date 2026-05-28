import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN ?? "";

function povertyColor(rate: number | null, opacity: number): Cesium.Color {
  const t = Math.min((rate ?? 50) / 80, 1);
  return new Cesium.Color(0.6 + 0.4 * t, 0.6 * (1 - t), 0.1, opacity);
}

function conflictColor(fatalities: number): Cesium.Color {
  if (fatalities > 15) return Cesium.Color.RED.withAlpha(0.95);
  if (fatalities > 5) return Cesium.Color.ORANGE.withAlpha(0.9);
  return Cesium.Color.YELLOW.withAlpha(0.85);
}

interface Props {
  onCountryClick: (f: PovertyFeature) => void;
}

export default function Globe({ onCountryClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const nlLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const ndviLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const settlementsLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const infraLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const waterLayerRef = useRef<Cesium.ImageryLayer | null>(null);

  const { layers, povertyFeatures, conflictEvents, flyTo, setFlyTo } = useGlobeStore();

  // ── Init viewer once ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    const creditDiv = document.createElement("div");

    const viewer = new Cesium.Viewer(containerRef.current, {
      timeline: false,
      animation: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      infoBox: false,
      selectionIndicator: false,
      creditContainer: creditDiv,
      scene3DOnly: true,
    });

    // Remove default Bing Maps/Ion layer — we manage all imagery ourselves
    viewer.imageryLayers.removeAll();

    // Pure black ocean/land base — gives the "Earth at night from space" look
    viewer.scene.globe.baseColor = Cesium.Color.BLACK;
    viewer.scene.backgroundColor = Cesium.Color.BLACK;

    // Subtle country-borders-only tile layer (very low opacity, just for geography reference)
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
        credit: "© CartoDB",
        minimumLevel: 0,
        maximumLevel: 19,
      })
    );
    viewer.imageryLayers.get(0).alpha = 0.25; // barely visible — just ghost borders

    // Focus on Africa
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(20, 5, 12_000_000),
    });

    // Click handler
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((e: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
      const picked = viewer.scene.pick(e.position);
      if (Cesium.defined(picked) && picked.id?.properties?.featureData) {
        const f = picked.id.properties.featureData.getValue(
          Cesium.JulianDate.now()
        ) as PovertyFeature;
        onCountryClick(f);
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    viewerRef.current = viewer;

    return () => {
      handler.destroy();
      if (!viewer.isDestroyed()) viewer.destroy();
      viewerRef.current = null;
    };
    // onCountryClick intentionally omitted — stable reference expected
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Fly-to ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!flyTo || !viewerRef.current) return;
    viewerRef.current.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(flyTo[1], flyTo[0], flyTo[2]),
      duration: 2,
    });
    setFlyTo(null);
  }, [flyTo, setFlyTo]);

  // ── Nighttime lights layer ────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.nightlights.enabled) {
      if (!nlLayerRef.current) {
        // {TileMatrixSet} → "500m", {TileMatrix} → zoom level, {TileRow}/{TileCol} → tile coords
        nlLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.WebMapTileServiceImageryProvider({
            url: "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/VIIRS_SNPP_DayNightBand_ENCC/default/2023-10-01/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer: "VIIRS_SNPP_DayNightBand_ENCC",
            style: "default",
            format: "image/png",
            tileMatrixSetID: "500m",
            maximumLevel: 8,
            tilingScheme: new Cesium.GeographicTilingScheme(),
            credit: "NASA GSFC / GIBS",
          })
        );
      }
      nlLayerRef.current.alpha = layers.nightlights.opacity;
      nlLayerRef.current.show = true;
    } else if (nlLayerRef.current) {
      nlLayerRef.current.show = false;
    }
  }, [layers.nightlights]);

  // ── NDVI layer ────────────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.ndvi.enabled) {
      if (!ndviLayerRef.current) {
        ndviLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.WebMapTileServiceImageryProvider({
            url: "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_NDVI_8Day/default/2023-01-01/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer: "MODIS_Terra_NDVI_8Day",
            style: "default",
            format: "image/png",
            tileMatrixSetID: "250m",
            maximumLevel: 8,
            tilingScheme: new Cesium.GeographicTilingScheme(),
            credit: "NASA GSFC / GIBS",
          })
        );
      }
      ndviLayerRef.current.alpha = layers.ndvi.opacity;
      ndviLayerRef.current.show = true;
    } else if (ndviLayerRef.current) {
      ndviLayerRef.current.show = false;
    }
  }, [layers.ndvi]);

  // ── Settlement density layer (CartoDB Light — grey urban footprints) ───────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.settlements.enabled) {
      if (!settlementsLayerRef.current) {
        settlementsLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
            credit: "© CartoDB",
            maximumLevel: 19,
          })
        );
      }
      settlementsLayerRef.current.alpha = layers.settlements.opacity;
      settlementsLayerRef.current.show = true;
    } else if (settlementsLayerRef.current) {
      settlementsLayerRef.current.show = false;
    }
  }, [layers.settlements]);

  // ── Infrastructure layer (HOT OpenStreetMap — roads, hospitals, schools) ───
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.infrastructure.enabled) {
      if (!infraLayerRef.current) {
        infraLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            credit: "© OpenStreetMap contributors, HOT",
            maximumLevel: 18,
          })
        );
      }
      infraLayerRef.current.alpha = layers.infrastructure.opacity;
      infraLayerRef.current.show = true;
    } else if (infraLayerRef.current) {
      infraLayerRef.current.show = false;
    }
  }, [layers.infrastructure]);

  // ── Water access layer (JRC Global Surface Water — permanent water bodies) ─
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.water.enabled) {
      if (!waterLayerRef.current) {
        waterLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://storage.googleapis.com/global-surface-water/tiles/occurrence/{z}/{x}/{y}.png",
            credit: "© EC JRC / Google",
            maximumLevel: 13,
          })
        );
      }
      waterLayerRef.current.alpha = layers.water.opacity;
      waterLayerRef.current.show = true;
    } else if (waterLayerRef.current) {
      waterLayerRef.current.show = false;
    }
  }, [layers.water]);

  // ── Poverty + conflict entities ───────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.entities.removeAll();

    if (layers.poverty.enabled) {
      povertyFeatures.forEach((f) => {
        viewer.entities.add({
          position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat),
          point: {
            pixelSize: 5,
            color: povertyColor(f.poverty_rate, layers.poverty.opacity),
            outlineWidth: 0,
            scaleByDistance: new Cesium.NearFarScalar(8e5, 1.8, 8e6, 0.4),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          properties: new Cesium.PropertyBag({ featureData: f }),
        });
      });
    }

    if (layers.conflict.enabled) {
      conflictEvents.forEach((ev) => {
        viewer.entities.add({
          position: Cesium.Cartesian3.fromDegrees(ev.lon, ev.lat),
          point: {
            pixelSize: 9,
            color: conflictColor(ev.fatalities).withAlpha(layers.conflict.opacity),
            outlineColor: Cesium.Color.RED.withAlpha(0.5),
            outlineWidth: 2,
            scaleByDistance: new Cesium.NearFarScalar(1e6, 1.8, 8e6, 0.6),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        });
      });
    }
  }, [povertyFeatures, conflictEvents, layers.poverty, layers.conflict]);

  return (
    <div
      ref={containerRef}
      style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
    />
  );
}
