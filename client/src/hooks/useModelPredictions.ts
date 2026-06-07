import { useEffect, useRef } from "react";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

function makeFlatTrend(v: number): number[] {
  return Array(10).fill(parseFloat(v.toFixed(3)));
}

/**
 * Derive SDG 1/7/11 scores and composite from the geojson feature fields.
 *
 * SDG 1  — DHS wealth index [-2, +2] → [0, 100]
 * SDG 7  — VIIRS NTL (nW/cm²/sr), threshold 1.0 → [0, 100]
 * SDG 11 — S2 brightness proxy (derived from NDBI + NTL), threshold 0.3 → [0, 100]
 * Composite — weighted mean (SDG1=1.0, SDG7=0.5, SDG11=0.5)
 */
function computeSdgScores(wi: number, ntl: number, ndbi: number) {
  const sdg1 = Math.round(Math.min(Math.max((wi + 2) / 4, 0), 1) * 1000) / 10;
  const sdg7 = Math.round(Math.min(ntl / 1.0, 1) * 1000) / 10;
  const brightness = 0.15 + ndbi * 0.25 + ntl * 0.05;
  const sdg11 = Math.round(Math.min(Math.max(brightness / 0.3, 0), 1) * 1000) / 10;
  const composite = Math.round(((sdg1 * 1.0 + sdg7 * 0.5 + sdg11 * 0.5) / 2.0) * 10) / 10;
  return { sdg1, sdg7, sdg11, composite };
}

export function useModelPredictions() {
  const setPovertyFeatures = useGlobeStore((s) => s.setPovertyFeatures);
  // Wait for World Bank data to load first before merging
  const mergedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      // Small delay so World Bank hook fires first
      await new Promise((r) => setTimeout(r, 1500));
      if (cancelled) return;
      try {
        // Served as static asset from /public — no Render dependency
        const res = await fetch("/predictions.geojson");
        if (!res.ok || cancelled) return;
        const data = await res.json();

        const clusterFeatures: PovertyFeature[] = (data.features ?? []).map(
          (f: {
            geometry: { coordinates: [number, number] };
            properties: {
              country: string; wealth_index: number; composite_score?: number;
              adm1_name?: string; region_name?: string; urban_rural?: string;
              iso3?: string;
              ntl_latest?: number; ntl_trend?: number;
              ndvi_latest?: number; ndbi_latest?: number;
              sdg1_score?: number; sdg7_score?: number; sdg11_score?: number;
              uncertainty?: number;
            };
          }) => {
            const [lon, lat] = f.geometry.coordinates;
            const wi   = f.properties.wealth_index ?? 0;
            const ntl  = f.properties.ntl_latest   ?? 0;
            const ndbi = f.properties.ndbi_latest  ?? 0;
            const poverty_rate = Math.max(0, Math.min(100, 50 - wi * 25));
            const adm1  = f.properties.adm1_name  ?? "";
            const urban = f.properties.urban_rural ?? "";
            const iso3  = f.properties.iso3 ?? "";
            // Encode place info into iso3 field — decoded by RegionPopup
            const placeKey = adm1 ? `CLUSTER|${adm1}|${urban}` : `COUNTRY|${iso3}`;
            const sdg = computeSdgScores(wi, ntl, ndbi);
            return {
              country:     f.properties.country,
              iso3:        placeKey,
              lat, lon, poverty_rate,
              hdi:         (wi + 2) / 4,
              year:        2023,
              ntl_trend:   makeFlatTrend(Math.max(0, wi / 4 + 0.25)),
              ndvi_trend:  makeFlatTrend(0.5),
              // Real satellite signals embedded by phase2_predict.py
              ntl_latest:   f.properties.ntl_latest,
              ntl_yr_trend: f.properties.ntl_trend,
              ndvi_latest:  f.properties.ndvi_latest,
              ndbi_latest:  f.properties.ndbi_latest,
              // SDG scores — computed from available signals
              sdg1_score:      sdg.sdg1,
              sdg7_score:      sdg.sdg7,
              sdg11_score:     sdg.sdg11,
              composite_score: sdg.composite,
              uncertainty:     f.properties.uncertainty ?? null,
            };
          }
        );

        if (clusterFeatures.length === 0 || cancelled) return;

        // Keep World Bank country-level dots for countries NOT in our cluster data
        const clusterCountries = new Set(clusterFeatures.map((f) => f.country));
        const existing = useGlobeStore.getState().povertyFeatures;
        const otherCountries = existing.filter((f) => !clusterCountries.has(f.country));

        mergedRef.current = true;
        setPovertyFeatures([...clusterFeatures, ...otherCountries]);
      } catch {
        // fall through — World Bank data stays
      }
    }
    load();
    return () => { cancelled = true; };
  }, [setPovertyFeatures]);
}
