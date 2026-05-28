import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

// ── Keplerian orbit propagator (no external dependency) ───────────────────────
const MU = 3.986e14;          // Earth gravitational parameter (m³/s²)
const R_EARTH = 6.371e6;      // Earth radius (m)
const OMEGA_E = 7.2921150e-5; // Earth rotation rate (rad/s)
const J2000_S = 946728000;    // Unix timestamp of J2000.0 epoch (s)

interface Orbit {
  label: string; altKm: number; incDeg: number; raanDeg: number; nu0Deg: number;
}

const SAT_ORBITS: Orbit[] = [
  { label: "ISS",          altKm: 408, incDeg: 51.6, raanDeg:   0, nu0Deg:   0 },
  { label: "Tiangong",     altKm: 390, incDeg: 41.5, raanDeg:  36, nu0Deg:  22 },
  { label: "Sentinel-2A",  altKm: 786, incDeg: 98.6, raanDeg:  72, nu0Deg:  44 },
  { label: "Sentinel-2B",  altKm: 786, incDeg: 98.6, raanDeg: 108, nu0Deg:  66 },
  { label: "Landsat-8",    altKm: 705, incDeg: 98.2, raanDeg: 144, nu0Deg:  88 },
  { label: "Landsat-9",    altKm: 705, incDeg: 98.2, raanDeg: 180, nu0Deg: 110 },
  { label: "VIIRS / SNPP", altKm: 824, incDeg: 98.7, raanDeg: 216, nu0Deg: 132 },
  { label: "MODIS Terra",  altKm: 705, incDeg: 98.2, raanDeg: 252, nu0Deg: 154 },
  { label: "MODIS Aqua",   altKm: 705, incDeg: 98.2, raanDeg: 288, nu0Deg: 176 },
  { label: "Hubble",       altKm: 540, incDeg: 28.5, raanDeg: 324, nu0Deg: 198 },
];

function orbitCartesian(orbit: Orbit, dtSec: number, epochMs: number): Cesium.Cartesian3 {
  const a = R_EARTH + orbit.altKm * 1e3;
  const n = Math.sqrt(MU / (a * a * a));
  const inc  = orbit.incDeg  * Math.PI / 180;
  const raan = orbit.raanDeg * Math.PI / 180;
  const nu   = (orbit.nu0Deg * Math.PI / 180) + n * dtSec;

  // Perifocal → ECI (circular orbit, argument of periapsis = 0)
  const xP = a * Math.cos(nu), yP = a * Math.sin(nu);
  const xE = xP * Math.cos(raan) - yP * Math.cos(inc) * Math.sin(raan);
  const yE = xP * Math.sin(raan) + yP * Math.cos(inc) * Math.cos(raan);
  const zE = yP * Math.sin(inc);

  // ECI → ECEF (rotate by GMST)
  const t0   = epochMs / 1000 - J2000_S;
  const gmst = (280.46061837 + 360.98564736629 * t0 / 86400) * Math.PI / 180
             + OMEGA_E * dtSec;
  return new Cesium.Cartesian3(
    xE * Math.cos(gmst) + yE * Math.sin(gmst),
   -xE * Math.sin(gmst) + yE * Math.cos(gmst),
    zE
  );
}

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

  // ── Live satellite tracking (Keplerian, no external fetch) ──────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const DURATION = 5760;
    const STEP     = 60;
    const epochMs  = Date.now();
    const startJD  = Cesium.JulianDate.fromDate(new Date(epochMs));
    const stopJD   = Cesium.JulianDate.addSeconds(startJD, DURATION, new Cesium.JulianDate());

    viewer.clock.startTime     = startJD.clone();
    viewer.clock.stopTime      = stopJD.clone();
    viewer.clock.currentTime   = startJD.clone();
    viewer.clock.clockRange    = Cesium.ClockRange.LOOP_STOP;
    viewer.clock.multiplier    = 60;
    viewer.clock.shouldAnimate = true;

    const satEntities: Cesium.Entity[] = [];
    for (const orbit of SAT_ORBITS) {
      const pos = new Cesium.SampledPositionProperty();
      pos.setInterpolationOptions({
        interpolationDegree: 5,
        interpolationAlgorithm: Cesium.LagrangePolynomialApproximation,
      });
      for (let dt = 0; dt <= DURATION; dt += STEP) {
        pos.addSample(
          Cesium.JulianDate.addSeconds(startJD, dt, new Cesium.JulianDate()),
          orbitCartesian(orbit, dt, epochMs)
        );
      }
      satEntities.push(viewer.entities.add({
        availability: new Cesium.TimeIntervalCollection([
          new Cesium.TimeInterval({ start: startJD, stop: stopJD }),
        ]),
        position: pos,
        point: {
          pixelSize: 7,
          color: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.CYAN,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        path: {
          resolution: 60,
          material: new Cesium.PolylineGlowMaterialProperty({
            glowPower: 0.2,
            color: Cesium.Color.CYAN.withAlpha(0.55),
          }),
          width: 1.5,
          leadTime: 0,
          trailTime: 720,
        },
      }));
    }

    return () => {
      const v = viewerRef.current;
      if (v && !v.isDestroyed()) {
        satEntities.forEach((e) => v.entities.remove(e));
        v.clock.shouldAnimate = false;
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
