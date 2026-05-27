import { useEffect } from "react";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

// Sub-Saharan Africa ISO3 codes with approximate centroids
const SSA_COUNTRIES: { iso3: string; name: string; lat: number; lon: number }[] = [
  { iso3: "NGA", name: "Nigeria",           lat:  9.08,  lon:  8.68 },
  { iso3: "ETH", name: "Ethiopia",          lat:  9.15,  lon: 40.49 },
  { iso3: "COD", name: "DR Congo",          lat: -4.04,  lon: 21.76 },
  { iso3: "KEN", name: "Kenya",             lat: -0.02,  lon: 37.91 },
  { iso3: "TZA", name: "Tanzania",          lat: -6.37,  lon: 34.89 },
  { iso3: "MOZ", name: "Mozambique",        lat:-18.67,  lon: 35.53 },
  { iso3: "GHA", name: "Ghana",             lat:  7.95,  lon:  1.02 },
  { iso3: "UGA", name: "Uganda",            lat:  1.37,  lon: 32.29 },
  { iso3: "CMR", name: "Cameroon",          lat:  3.85,  lon: 11.50 },
  { iso3: "AGO", name: "Angola",            lat:-11.20,  lon: 17.87 },
  { iso3: "ZMB", name: "Zambia",            lat:-13.13,  lon: 27.85 },
  { iso3: "ZWE", name: "Zimbabwe",          lat:-19.02,  lon: 29.15 },
  { iso3: "MWI", name: "Malawi",            lat:-13.25,  lon: 34.30 },
  { iso3: "SEN", name: "Senegal",           lat: 14.50,  lon:-14.45 },
  { iso3: "MLI", name: "Mali",              lat: 17.57,  lon:  -3.99 },
  { iso3: "BFA", name: "Burkina Faso",      lat: 12.36,  lon:  -1.53 },
  { iso3: "RWA", name: "Rwanda",            lat: -1.94,  lon: 29.87 },
  { iso3: "NER", name: "Niger",             lat: 17.61,  lon:   8.08 },
  { iso3: "TCD", name: "Chad",              lat: 15.45,  lon: 18.73 },
  { iso3: "MDG", name: "Madagascar",        lat:-18.77,  lon: 46.87 },
  { iso3: "ZAF", name: "South Africa",      lat:-28.47,  lon: 24.68 },
  { iso3: "SDN", name: "Sudan",             lat: 12.86,  lon: 30.22 },
  { iso3: "SOM", name: "Somalia",           lat:  5.15,  lon: 46.20 },
  { iso3: "GIN", name: "Guinea",            lat: 11.75,  lon:-15.45 },
  { iso3: "BWA", name: "Botswana",          lat:-22.33,  lon: 24.68 },
  { iso3: "NAM", name: "Namibia",           lat:-22.96,  lon: 18.49 },
  { iso3: "SLE", name: "Sierra Leone",      lat:  8.46,  lon:-11.78 },
  { iso3: "TGO", name: "Togo",              lat:  8.62,  lon:   0.82 },
  { iso3: "BEN", name: "Benin",             lat:  9.31,  lon:   2.32 },
  { iso3: "HTI", name: "Haiti",             lat: 18.97,  lon: -72.29 },
];

// Sparkline data shape: 10 year points
function makeFakeTrend(base: number, noise = 0.05): number[] {
  const arr: number[] = [];
  let v = base;
  for (let i = 0; i < 10; i++) {
    v = Math.max(0, Math.min(1, v + (Math.random() - 0.5) * noise));
    arr.push(parseFloat(v.toFixed(3)));
  }
  return arr;
}

export function useWorldBank(year: number) {
  const setPovertyFeatures = useGlobeStore((s) => s.setPovertyFeatures);

  useEffect(() => {
    async function load() {
      const features: PovertyFeature[] = [];

      // World Bank poverty headcount (SI.POV.DDAY) for each country
      const isos = SSA_COUNTRIES.map((c) => c.iso3).join(";");
      const url = `https://api.worldbank.org/v2/country/${isos}/indicator/SI.POV.DDAY?date=${year}&format=json&per_page=100`;
      const hdiUrl = `https://api.worldbank.org/v2/country/${isos}/indicator/SP.DYN.LE00.IN?date=${year}&format=json&per_page=100`;

      let povertyMap: Record<string, number | null> = {};
      let hdiMap: Record<string, number | null> = {};

      try {
        const [pRes, hRes] = await Promise.all([fetch(url), fetch(hdiUrl)]);
        const [pData, hData] = await Promise.all([pRes.json(), hRes.json()]);

        if (Array.isArray(pData) && pData[1]) {
          pData[1].forEach((row: { countryiso3code: string; value: number | null }) => {
            povertyMap[row.countryiso3code] = row.value;
          });
        }
        if (Array.isArray(hData) && hData[1]) {
          hData[1].forEach((row: { countryiso3code: string; value: number | null }) => {
            // Life expectancy as HDI proxy
            hdiMap[row.countryiso3code] = row.value != null ? row.value / 90 : null;
          });
        }
      } catch {
        // API unavailable — use demo values
      }

      SSA_COUNTRIES.forEach(({ iso3, name, lat, lon }) => {
        const pRate = povertyMap[iso3] ?? (30 + Math.random() * 40);
        const hdi = hdiMap[iso3] ?? (0.3 + Math.random() * 0.4);
        features.push({
          country: name,
          iso3,
          lat,
          lon,
          poverty_rate: pRate,
          hdi,
          year,
          ntl_trend: makeFakeTrend(0.2 + (1 - pRate / 100) * 0.5),
          ndvi_trend: makeFakeTrend(0.5 + Math.random() * 0.3, 0.08),
        });
      });

      setPovertyFeatures(features);
    }

    load();
  }, [year, setPovertyFeatures]);
}
