import { useEffect } from "react";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";
import { apiUrl } from "../lib/api";

function makeFlatTrend(v: number): number[] {
  return Array(10).fill(parseFloat(v.toFixed(3)));
}

export function useModelPredictions() {
  const setPovertyFeatures = useGlobeStore((s) => s.setPovertyFeatures);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(apiUrl("/api/predictions"));
        if (!res.ok) return;
        const data = await res.json();
        const features: PovertyFeature[] = (data.features ?? []).map(
          (f: {
            geometry: { coordinates: [number, number] };
            properties: {
              country: string;
              wealth_index: number;
              composite_score: number;
            };
          }) => {
            const [lon, lat] = f.geometry.coordinates;
            const wi = f.properties.wealth_index ?? 0;
            // Convert wealth index [-2, +2] to poverty rate [0, 100]
            // wi=-2 → very poor → poverty_rate≈100; wi=+2 → wealthy → poverty_rate≈0
            const poverty_rate = Math.max(0, Math.min(100, 50 - wi * 25));
            return {
              country: f.properties.country,
              iso3: "",
              lat,
              lon,
              poverty_rate,
              hdi: f.properties.composite_score != null
                ? f.properties.composite_score / 100
                : null,
              year: 2023,
              ntl_trend: makeFlatTrend(Math.max(0, wi / 4 + 0.25)),
              ndvi_trend: makeFlatTrend(0.5),
            };
          }
        );
        if (features.length > 0) setPovertyFeatures(features);
      } catch {
        // fall through — World Bank hook already set features
      }
    }
    load();
  }, [setPovertyFeatures]);
}
