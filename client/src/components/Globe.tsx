import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import * as satellite from "satellite.js";
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
  const povertyPointsRef = useRef<Cesium.PointPrimitiveCollection | null>(null);
  const conflictPointsRef = useRef<Cesium.PointPrimitiveCollection | null>(null);

  const layers = useGlobeStore((s) => s.layers);
  const flyTo = useGlobeStore((s) => s.flyTo);
  const setFlyTo = useGlobeStore((s) => s.setFlyTo);

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

    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#05080f");
    viewer.scene.backgroundColor = Cesium.Color.BLACK;

    // Esri World Imagery — real satellite photos of Earth
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        credit: "© Esri, Maxar, Earthstar Geographics",
        maximumLevel: 19,
      })
    );
    // Subtle dark overlay to keep the dashboard aesthetic without washing out imagery
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
        credit: "© CartoDB",
        maximumLevel: 19,
      })
    );
    viewer.imageryLayers.get(1).alpha = 0.45;

    // Focus on Africa
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(20, 5, 12_000_000),
    });

    // Click handler — works with PointPrimitive ids (stored directly as PovertyFeature)
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((e: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
      const picked = viewer.scene.pick(e.position);
      if (Cesium.defined(picked) && picked.id && (picked.id as PovertyFeature).iso3) {
        onCountryClick(picked.id as PovertyFeature);
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

  // ── Poverty + conflict point primitives ──────────────────────────────────
  // PointPrimitiveCollection is GPU-batched — handles 3000+ points without
  // the entity API's per-add event overhead that corrupts the pick buffer.
  useEffect(() => {
    function rebuild(state: ReturnType<typeof useGlobeStore.getState>) {
      const viewer = viewerRef.current;
      if (!viewer) return;

      // Swap poverty collection
      if (povertyPointsRef.current) {
        viewer.scene.primitives.remove(povertyPointsRef.current);
        povertyPointsRef.current = null;
      }
      if (state.layers.poverty.enabled && state.povertyFeatures.length > 0) {
        const col = new Cesium.PointPrimitiveCollection();
        state.povertyFeatures.forEach((f) => {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat),
            color: povertyColor(f.poverty_rate, state.layers.poverty.opacity),
            pixelSize: 5,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(8e5, 1.8, 8e6, 0.4),
            id: f,
          });
        });
        viewer.scene.primitives.add(col);
        povertyPointsRef.current = col;
      }

      // Swap conflict collection
      if (conflictPointsRef.current) {
        viewer.scene.primitives.remove(conflictPointsRef.current);
        conflictPointsRef.current = null;
      }
      if (state.layers.conflict.enabled && state.conflictEvents.length > 0) {
        const col = new Cesium.PointPrimitiveCollection();
        state.conflictEvents.forEach((ev) => {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(ev.lon, ev.lat),
            color: conflictColor(ev.fatalities).withAlpha(state.layers.conflict.opacity),
            pixelSize: 9,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(1e6, 1.8, 8e6, 0.6),
          });
        });
        viewer.scene.primitives.add(col);
        conflictPointsRef.current = col;
      }
    }

    let prevPoverty = useGlobeStore.getState().povertyFeatures;
    let prevConflict = useGlobeStore.getState().conflictEvents;
    let prevLayers = useGlobeStore.getState().layers;

    rebuild(useGlobeStore.getState());

    const unsub = useGlobeStore.subscribe((state) => {
      if (
        state.povertyFeatures !== prevPoverty ||
        state.conflictEvents !== prevConflict ||
        state.layers !== prevLayers
      ) {
        prevPoverty = state.povertyFeatures;
        prevConflict = state.conflictEvents;
        prevLayers = state.layers;
        rebuild(state);
      }
    });

    return unsub;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Live satellite tracking (Cesium clock-driven, smooth interpolation) ─────
  useEffect(() => {
    interface TLE { name: string; line1: string; line2: string }
    const satEntities: Cesium.Entity[] = [];
    let cancelled = false;

    async function loadAndRender() {
      const tles: TLE[] = [];
      // Try two groups — stations (ISS, Tiangong) then visual (100 brightest)
      const GROUPS = ["stations", "visual"];
      for (const group of GROUPS) {
        try {
          const res = await fetch(
            `https://celestrak.org/NOSTR/GP.php?GROUP=${group}&FORMAT=tle`,
            { cache: "no-store" }
          );
          if (!res.ok) continue;
          const text = await res.text();
          const lines = text.trim().split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
          for (let i = 0; i + 2 < lines.length; i += 3) {
            tles.push({ name: lines[i], line1: lines[i + 1], line2: lines[i + 2] });
            if (tles.length >= 25) break;
          }
          if (tles.length > 0) break;
        } catch { continue; }
      }
      if (tles.length === 0) return;

      if (cancelled) return;
      const viewer = viewerRef.current;
      if (!viewer) return;

      // One full LEO orbit = 90 minutes; sample every 60 s → 91 position samples
      const DURATION = 5400;
      const STEP     = 60;
      const startJD  = Cesium.JulianDate.fromDate(new Date());
      const stopJD   = Cesium.JulianDate.addSeconds(startJD, DURATION, new Cesium.JulianDate());

      // Drive Cesium's clock silently (no timeline UI) — 60 s of orbit per real second
      viewer.clock.startTime    = startJD.clone();
      viewer.clock.stopTime     = stopJD.clone();
      viewer.clock.currentTime  = startJD.clone();
      viewer.clock.clockRange   = Cesium.ClockRange.LOOP_STOP;
      viewer.clock.multiplier   = 60;
      viewer.clock.shouldAnimate = true;

      for (const tle of tles) {
        if (cancelled) break;
        try {
          const satrec = satellite.twoline2satrec(tle.line1, tle.line2);
          const posProperty = new Cesium.SampledPositionProperty();
          posProperty.setInterpolationOptions({
            interpolationDegree: 5,
            interpolationAlgorithm: Cesium.LagrangePolynomialApproximation,
          });

          for (let dt = 0; dt <= DURATION; dt += STEP) {
            const jd     = Cesium.JulianDate.addSeconds(startJD, dt, new Cesium.JulianDate());
            const jsDate = Cesium.JulianDate.toDate(jd);
            const pv     = satellite.propagate(satrec, jsDate);
            if (!pv || !pv.position || typeof pv.position === "boolean") continue;
            const gmst = satellite.gstime(jsDate);
            const geo  = satellite.eciToGeodetic(pv.position as satellite.EciVec3<number>, gmst);
            posProperty.addSample(
              jd,
              Cesium.Cartesian3.fromDegrees(
                geo.longitude * (180 / Math.PI),
                geo.latitude  * (180 / Math.PI),
                geo.height * 1000
              )
            );
          }

          const entity = viewer.entities.add({
            availability: new Cesium.TimeIntervalCollection([
              new Cesium.TimeInterval({ start: startJD, stop: stopJD }),
            ]),
            position: posProperty,
            point: {
              pixelSize: 8,
              color: Cesium.Color.WHITE,
              outlineColor: Cesium.Color.CYAN,
              outlineWidth: 2,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
            path: {
              resolution: 60,
              // glowing cyan trail behind each satellite
              material: new Cesium.PolylineGlowMaterialProperty({
                glowPower: 0.15,
                color: Cesium.Color.CYAN.withAlpha(0.5),
              }),
              width: 1.5,
              leadTime: 0,
              trailTime: 600, // show last 10 minutes of orbit as a trail
            },
          });
          satEntities.push(entity);
        } catch { continue; }
      }
    }

    loadAndRender();

    return () => {
      cancelled = true;
      const viewer = viewerRef.current;
      if (viewer && !viewer.isDestroyed()) {
        satEntities.forEach((e) => viewer.entities.remove(e));
        viewer.clock.shouldAnimate = false;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
    />
  );
}
