import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";
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
        // Clicked empty space — dismiss all popups
        store.setSelected(null);
        store.setSelectedSatellite(null);
        return;
      }

      // PointPrimitive ids are PovertyFeature objects (have .iso3 string)
      if (typeof (picked.id as PovertyFeature).iso3 === "string") {
        onCountryClick(picked.id as PovertyFeature);
        store.setSelectedSatellite(null);
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

  // ── Nighttime lights layer ────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.nightlights.enabled) {
      if (!nlLayerRef.current) {
        nlLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            // NASA GIBS VIIRS CityLights — static product (no time param), WebMercator.
            // URL confirmed 200 OK. {z}/{y}/{x} = TileMatrix/TileRow/TileCol per GIBS WMTS spec.
            url: "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_CityLights_2012/default/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg",
            maximumLevel: 8,
            credit: "NASA / VIIRS City Lights",
          })
        );
        nlLayerRef.current.colorToAlpha = new Cesium.Color(0.0, 0.0, 0.0, 1.0);
        nlLayerRef.current.colorToAlphaThreshold = 0.05;
      }
      nlLayerRef.current.alpha = layers.nightlights.opacity;
      nlLayerRef.current.show  = true;
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
        // GIBS EPSG:4326 250m tiles are 512×512px — must set tileWidth/tileHeight or Cesium
        // requests wrong tile coordinates and gets 404s for half the grid positions.
        ndviLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            // Date must be within GIBS retention window (starts 2025-02-12 for this product).
            // 2023-06-01 was outside the window and returned 404 for every tile.
            url: "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_NDVI_8Day/default/2026-05-01/250m/{z}/{y}/{x}.png",
            tilingScheme: new Cesium.GeographicTilingScheme(),
            tileWidth: 512,
            tileHeight: 512,
            maximumLevel: 8,
            credit: "NASA GSFC / GIBS — MODIS Terra NDVI",
          })
        );
      }
      ndviLayerRef.current.alpha = layers.ndvi.opacity;
      ndviLayerRef.current.show  = true;
    } else if (ndviLayerRef.current) {
      ndviLayerRef.current.show = false;
    }
  }, [layers.ndvi]);

  // ── Settlement density layer ───────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layers.settlements.enabled) {
      if (!settlementsLayerRef.current) {
        // CartoDB Positron (light) contrasts with the dark satellite base — urban areas
        // appear white/grey showing the street+building footprint (settlement density proxy).
        // The previous dark_matter_nolabels was identical to the base overlay and invisible.
        settlementsLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
            credit: "© CartoDB Positron",
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
            url: "https://tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
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
        // MODIS Water Mask (GIBS EPSG:4326, 250m, 512×512 tiles) — static product, URL 200 OK.
        // JRC_Global_Surface_Water_Recurrence is not served in the GIBS EPSG:4326 endpoint.
        // Previous CartoDB Voyager nolabels was nearly indistinguishable from the base overlay.
        waterLayerRef.current = viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Water_Mask/default/2019-01-01/250m/{z}/{y}/{x}.png",
            tilingScheme: new Cesium.GeographicTilingScheme(),
            tileWidth: 512,
            tileHeight: 512,
            maximumLevel: 8,
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
            pixelSize: 9,
            scaleByDistance: new Cesium.NearFarScalar(1e6, 1.8, 8e6, 0.6),
          });
        });
        viewer.scene.primitives.add(col);
        conflictPointsRef.current = col;
      }
    }

    let prevPoverty  = useGlobeStore.getState().povertyFeatures;
    let prevConflict = useGlobeStore.getState().conflictEvents;
    let prevLayers   = useGlobeStore.getState().layers;

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
