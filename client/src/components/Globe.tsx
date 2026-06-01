import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature, ConflictEvent } from "../store/globeStore";
import { SAT_ORBITS, orbitXYZ, MU, R_EARTH } from "../utils/orbitPropagator";
import type { SatelliteInfo } from "../utils/orbitPropagator";

Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN ?? "";

function povertyColor(rate: number | null, opacity: number): Cesium.Color {
  const t = Math.min((rate ?? 50) / 80, 1);
  return new Cesium.Color(0.6 + 0.4 * t, 0.6 * (1 - t), 0.1, opacity);
}

function conflictColor(fatalities: number): Cesium.Color {
  if (fatalities > 15) return Cesium.Color.RED.withAlpha(0.95);
  if (fatalities > 5)  return Cesium.Color.ORANGE.withAlpha(0.9);
  return Cesium.Color.YELLOW.withAlpha(0.85);
}

interface Props {
  onCountryClick: (f: PovertyFeature) => void;
}

export default function Globe({ onCountryClick }: Props) {
  const containerRef      = useRef<HTMLDivElement>(null);
  const viewerRef         = useRef<Cesium.Viewer | null>(null);
  const nlLayerRef        = useRef<Cesium.ImageryLayer | null>(null);
  const ndviLayerRef      = useRef<Cesium.ImageryLayer | null>(null);
  const settlementsLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const infraLayerRef     = useRef<Cesium.ImageryLayer | null>(null);
  const waterLayerRef     = useRef<Cesium.ImageryLayer | null>(null);
  const povertyPointsRef  = useRef<Cesium.PointPrimitiveCollection | null>(null);
  const conflictPointsRef = useRef<Cesium.PointPrimitiveCollection | null>(null);
  // satellite entity id → SatelliteInfo lookup for click detection
  const satInfoRef        = useRef<Map<string, SatelliteInfo>>(new Map());
  // epoch used for the current satellite animation (shared with click handler)
  const satEpochMsRef     = useRef<number>(0);
  // currently displayed full-orbit groundtrack entity
  const groundtrackRef    = useRef<Cesium.Entity | null>(null);
  // currently highlighted satellite entity (reset color on deselect)
  const highlightedEntityRef = useRef<Cesium.Entity | null>(null);

  const layers   = useGlobeStore((s) => s.layers);
  const flyTo    = useGlobeStore((s) => s.flyTo);
  const setFlyTo = useGlobeStore((s) => s.setFlyTo);
  const year     = useGlobeStore((s) => s.year);

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

    // Render at native device pixel density — fixes pixelation on retina/HiDPI displays.
    // Without this, Cesium renders at 1× CSS pixels; on a 2× DPR screen every pixel
    // covers 4 physical pixels, producing a blurry 240p-style appearance.
    viewer.resolutionScale = window.devicePixelRatio;

    viewer.imageryLayers.removeAll();
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#0a1628");
    viewer.scene.backgroundColor = Cesium.Color.BLACK;

    // EOX S2Cloudless 2021 — real Sentinel-2 satellite imagery, cloud-free annual composite.
    // Standard XYZ/WebMercator tiles: no projection mismatch, loads identically to CartoDB.
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg",
        credit: "EOX IT Services — Sentinel-2 cloudless 2021",
        maximumLevel: 14,
      })
    );
    // Dark overlay to keep dashboard contrast for data points
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
        credit: "© CartoDB",
        maximumLevel: 19,
      })
    );
    viewer.imageryLayers.get(1).alpha = 0.4;

    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(20, 5, 12_000_000),
    });

    // Click handler — detects both PointPrimitive (poverty/conflict) and Entity (satellite)
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((e: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
      const picked = viewer.scene.pick(e.position);
      const store  = useGlobeStore.getState();

      if (!Cesium.defined(picked) || !picked.id) {
        store.setSelected(null);
        store.setSelectedSatellite(null);
        store.setSelectedConflict(null);
        return;
      }

      // ConflictEvent points have .event_type string
      if (typeof (picked.id as ConflictEvent).event_type === "string") {
        store.setSelectedConflict(picked.id as ConflictEvent);
        store.setSelected(null);
        store.setSelectedSatellite(null);
        return;
      }

      // PovertyFeature points have .iso3 string
      if (typeof (picked.id as PovertyFeature).iso3 === "string") {
        onCountryClick(picked.id as PovertyFeature);
        store.setSelectedSatellite(null);
        store.setSelectedConflict(null);
        return;
      }

      // Entity ids are Cesium.Entity instances (have .id UUID string)
      const entityId = typeof (picked.id as Cesium.Entity).id === "string"
        ? (picked.id as Cesium.Entity).id
        : null;
      if (entityId && satInfoRef.current.has(entityId)) {
        store.setSelectedSatellite(satInfoRef.current.get(entityId)!);
        store.setSatEpochMs(satEpochMsRef.current);
        store.setSelected(null);
        store.setSelectedConflict(null);
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    viewerRef.current = viewer;

    return () => {
      handler.destroy();
      if (!viewer.isDestroyed()) viewer.destroy();
      viewerRef.current = null;
    };
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

  // ── Nighttime lights layer — year-responsive ──────────────────────────────
  // GIBS VIIRS_Black_Marble only has two annual snapshots: 2012 and 2016.
  // Years ≤ 2014 → 2012 composite; years ≥ 2015 → 2016 composite.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Always remove + recreate so year changes take effect immediately.
    if (nlLayerRef.current) {
      viewer.imageryLayers.remove(nlLayerRef.current, true);
      nlLayerRef.current = null;
    }

    if (!layers.nightlights.enabled) return;

    const snap = year <= 2014 ? "2012-01-01" : "2016-01-01";
    nlLayerRef.current = viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/${snap}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png`,
        maximumLevel: 8,
        credit: "NASA / VIIRS Black Marble",
      })
    );
    nlLayerRef.current.colorToAlpha = new Cesium.Color(0.0, 0.0, 0.0, 1.0);
    nlLayerRef.current.colorToAlphaThreshold = 0.05;
    nlLayerRef.current.alpha = layers.nightlights.opacity;
  }, [layers.nightlights, year]);

  // ── NDVI layer — year-responsive ─────────────────────────────────────────
  // MODIS_Terra_L3_NDVI_Monthly covers 2000–present (confirmed 200 OK for all years).
  // June is used as the date — peak vegetation month for Sub-Saharan Africa.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (ndviLayerRef.current) {
      viewer.imageryLayers.remove(ndviLayerRef.current, true);
      ndviLayerRef.current = null;
    }

    if (!layers.ndvi.enabled) return;

    const clampedYear = Math.max(2000, Math.min(2023, year));
    ndviLayerRef.current = viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_L3_NDVI_Monthly/default/${clampedYear}-06-01/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png`,
        maximumLevel: 7,
        credit: `NASA GSFC / GIBS — MODIS NDVI ${clampedYear}`,
      })
    );
    ndviLayerRef.current.alpha = layers.ndvi.opacity;
  }, [layers.ndvi, year]);

  // ── Settlement density layer ───────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.settlements.enabled) {
      if (!settlementsLayerRef.current) {
        // Esri World Street Map — road density is a strong proxy for settlement density.
        // Urban cores show dense road grids; rural areas show sparse or no roads.
        // Esri tile format: tile/{z}/{y}/{x} (row/col, same convention as GIBS WMTS). Confirmed 200 OK.
        settlementsLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            credit: "© Esri / OpenStreetMap contributors",
            maximumLevel: 19,
          })
        );
      }
      settlementsLayerRef.current.alpha = layers.settlements.opacity;
      settlementsLayerRef.current.show  = true;
    } else if (settlementsLayerRef.current) {
      settlementsLayerRef.current.show = false;
    }
  }, [layers.settlements]);

  // ── Infrastructure layer ───────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.infrastructure.enabled) {
      if (!infraLayerRef.current) {
        infraLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            credit: "© OpenStreetMap contributors, HOT",
            maximumLevel: 18,
          })
        );
      }
      infraLayerRef.current.alpha = layers.infrastructure.opacity;
      infraLayerRef.current.show  = true;
    } else if (infraLayerRef.current) {
      infraLayerRef.current.show = false;
    }
  }, [layers.infrastructure]);

  // ── Water access layer ─────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.water.enabled) {
      if (!waterLayerRef.current) {
        // MODIS Water Mask — WebMercator epsg3857, no time param (static product), confirmed 200 OK.
        // Uses GoogleMapsCompatible_Level9 TileMatrixSet so no GeographicTilingScheme needed.
        waterLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Water_Mask/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png",
            maximumLevel: 9,
            credit: "NASA GSFC / GIBS — MODIS Water Mask",
          })
        );
      }
      waterLayerRef.current.alpha = layers.water.opacity;
      waterLayerRef.current.show  = true;
    } else if (waterLayerRef.current) {
      waterLayerRef.current.show = false;
    }
  }, [layers.water]);

  // ── Poverty + conflict point primitives ───────────────────────────────────
  useEffect(() => {
    let prevPoverty  = useGlobeStore.getState().povertyFeatures;
    let prevConflict = useGlobeStore.getState().conflictEvents;
    let prevLayers   = useGlobeStore.getState().layers;

    function rebuild(state: ReturnType<typeof useGlobeStore.getState>) {
      const viewer = viewerRef.current;
      if (!viewer) return;

      if (povertyPointsRef.current) {
        viewer.scene.primitives.remove(povertyPointsRef.current);
        povertyPointsRef.current = null;
      }
      if (state.layers.poverty.enabled && state.povertyFeatures.length > 0) {
        const col = new Cesium.PointPrimitiveCollection();
        state.povertyFeatures.forEach((f) => {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat, 20000),
            color: povertyColor(f.poverty_rate, state.layers.poverty.opacity),
            pixelSize: 9,
            outlineColor: Cesium.Color.WHITE.withAlpha(0.7),
            outlineWidth: 1.5,
            scaleByDistance: new Cesium.NearFarScalar(5e5, 2.0, 1e7, 0.7),
            id: f,
          });
        });
        viewer.scene.primitives.add(col);
        povertyPointsRef.current = col;
      }

      if (conflictPointsRef.current) {
        viewer.scene.primitives.remove(conflictPointsRef.current);
        conflictPointsRef.current = null;
      }
      if (state.layers.conflict.enabled && state.conflictEvents.length > 0) {
        const col = new Cesium.PointPrimitiveCollection();
        state.conflictEvents.forEach((ev) => {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(ev.lon, ev.lat, 20000),
            color: conflictColor(ev.fatalities).withAlpha(state.layers.conflict.opacity),
            pixelSize: 12,
            outlineColor: Cesium.Color.WHITE.withAlpha(0.6),
            outlineWidth: 1,
            scaleByDistance: new Cesium.NearFarScalar(5e5, 2.5, 1e7, 0.8),
            id: ev,
          });
        });
        viewer.scene.primitives.add(col);
        conflictPointsRef.current = col;
      }
    }

    rebuild(useGlobeStore.getState());

    const unsub = useGlobeStore.subscribe((state) => {
      if (
        state.povertyFeatures !== prevPoverty ||
        state.conflictEvents  !== prevConflict ||
        state.layers          !== prevLayers
      ) {
        prevPoverty  = state.povertyFeatures;
        prevConflict = state.conflictEvents;
        prevLayers   = state.layers;
        rebuild(state);
      }
    });

    return unsub;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Live satellite tracking (Keplerian, no external fetch) ────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const DURATION = 5760;
    const STEP     = 60;
    const epochMs  = Date.now();
    satEpochMsRef.current = epochMs;

    const startJD = Cesium.JulianDate.fromDate(new Date(epochMs));
    const stopJD  = Cesium.JulianDate.addSeconds(startJD, DURATION, new Cesium.JulianDate());

    viewer.clock.startTime     = startJD.clone();
    viewer.clock.stopTime      = stopJD.clone();
    viewer.clock.currentTime   = startJD.clone();
    viewer.clock.clockRange    = Cesium.ClockRange.LOOP_STOP;
    viewer.clock.multiplier    = 1;
    viewer.clock.shouldAnimate = true;

    satInfoRef.current.clear();
    const satEntities: Cesium.Entity[] = [];

    for (const orbit of SAT_ORBITS) {
      const pos = new Cesium.SampledPositionProperty();
      pos.setInterpolationOptions({
        interpolationDegree: 5,
        interpolationAlgorithm: Cesium.LagrangePolynomialApproximation,
      });
      for (let dt = 0; dt <= DURATION; dt += STEP) {
        const [x, y, z] = orbitXYZ(orbit, dt, epochMs);
        pos.addSample(
          Cesium.JulianDate.addSeconds(startJD, dt, new Cesium.JulianDate()),
          new Cesium.Cartesian3(x, y, z)
        );
      }

      const entity = viewer.entities.add({
        name: orbit.label,
        availability: new Cesium.TimeIntervalCollection([
          new Cesium.TimeInterval({ start: startJD, stop: stopJD }),
        ]),
        position: pos,
        point: {
          pixelSize: 7,
          color: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.CYAN,
          outlineWidth: 2,
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
        label: {
          text: orbit.label,
          font: "10px monospace",
          fillColor: Cesium.Color.CYAN,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(12, -4),
          translucencyByDistance: new Cesium.NearFarScalar(1e6, 1.0, 8e6, 0.0),
        },
      });

      satInfoRef.current.set(entity.id, orbit);
      satEntities.push(entity);
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

  // ── Full-orbit groundtrack when a satellite is selected ───────────────────
  useEffect(() => {
    function drawGroundtrack(sat: SatelliteInfo | null, epochMs: number) {
      const viewer = viewerRef.current;
      if (!viewer || viewer.isDestroyed()) return;

      // Reset previously highlighted entity colour
      if (highlightedEntityRef.current) {
        const prev = highlightedEntityRef.current;
        if (prev.point) {
          prev.point.pixelSize  = new Cesium.ConstantProperty(7);
          prev.point.color      = new Cesium.ConstantProperty(Cesium.Color.WHITE);
          prev.point.outlineColor = new Cesium.ConstantProperty(Cesium.Color.CYAN);
        }
        highlightedEntityRef.current = null;
      }

      // Remove old groundtrack
      if (groundtrackRef.current) {
        viewer.entities.remove(groundtrackRef.current);
        groundtrackRef.current = null;
      }

      if (!sat) return;

      // Highlight the selected satellite entity
      for (const [id, info] of satInfoRef.current.entries()) {
        if (info.label === sat.label) {
          const entity = viewer.entities.getById(id);
          if (entity?.point) {
            entity.point.pixelSize    = new Cesium.ConstantProperty(12);
            entity.point.color        = new Cesium.ConstantProperty(Cesium.Color.YELLOW);
            entity.point.outlineColor = new Cesium.ConstantProperty(Cesium.Color.ORANGE);
            highlightedEntityRef.current = entity;
          }
          break;
        }
      }

      // Draw one full orbit as a bright polyline
      const a      = R_EARTH + sat.altKm * 1e3;
      const T      = 2 * Math.PI * Math.sqrt((a * a * a) / MU);
      const dtStart = (Date.now() - epochMs) / 1000;
      const NSTEPS  = 200;
      const positions: Cesium.Cartesian3[] = [];
      for (let i = 0; i <= NSTEPS; i++) {
        const [x, y, z] = orbitXYZ(sat, dtStart + (T * i) / NSTEPS, epochMs);
        positions.push(new Cesium.Cartesian3(x, y, z));
      }

      groundtrackRef.current = viewer.entities.add({
        polyline: {
          positions,
          width: 2,
          material: new Cesium.ColorMaterialProperty(Cesium.Color.YELLOW.withAlpha(0.5)),
          arcType: Cesium.ArcType.NONE,
        },
      });
    }

    let prevSat = useGlobeStore.getState().selectedSatellite;

    const unsub = useGlobeStore.subscribe((state) => {
      if (state.selectedSatellite !== prevSat) {
        prevSat = state.selectedSatellite;
        drawGroundtrack(state.selectedSatellite, state.satEpochMs || satEpochMsRef.current);
      }
    });

    return () => {
      unsub();
      const viewer = viewerRef.current;
      if (viewer && !viewer.isDestroyed() && groundtrackRef.current) {
        viewer.entities.remove(groundtrackRef.current);
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
