import { useEffect, useRef } from "react";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

function makeFlatTrend(v: number): number[] {
  return Array(10).fill(parseFloat(v.toFixed(3)));
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
              country: string; wealth_index: number; composite_score: number;
              adm1_name?: string; region_name?: string; urban_rural?: string;
            };
          }) => {
            const [lon, lat] = f.geometry.coordinates;
            const wi = f.properties.wealth_index ?? 0;
            const poverty_rate = Math.max(0, Math.min(100, 50 - wi * 25));
            const adm1  = f.properties.adm1_name  ?? "";
            const urban = f.properties.urban_rural ?? "";
            // Encode place info into iso3 — decoded by RegionPopup
            const placeKey = `CLUSTER|${adm1}|${urban}`;
            return {
              country: f.properties.country,
              iso3: placeKey,
              lat, lon, poverty_rate,
              hdi: (wi + 2) / 4,
              year: 2023,
              ntl_trend: makeFlatTrend(Math.max(0, wi / 4 + 0.25)),
              ndvi_trend: makeFlatTrend(0.5),
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
