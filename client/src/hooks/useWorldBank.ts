import { useEffect } from "react";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

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

export function useWorldBank(year: number) {
  const setPovertyFeatures = useGlobeStore((s) => s.setPovertyFeatures);

  useEffect(() => {
    async function load() {
      const isos = SSA_COUNTRIES.map((c) => c.iso3).join(";");

      // Fetch current-year poverty + life expectancy (HDI proxy) + 10-year poverty history
      const povertyUrl = `https://api.worldbank.org/v2/country/${isos}/indicator/SI.POV.DDAY?date=${year}&format=json&per_page=100`;
      const hdiUrl = `https://api.worldbank.org/v2/country/${isos}/indicator/SP.DYN.LE00.IN?date=${year}&format=json&per_page=100`;
      const histUrl = `https://api.worldbank.org/v2/country/${isos}/indicator/SI.POV.DDAY?date=2014:2023&format=json&per_page=500&mrv=1`;

      let povertyMap: Record<string, number | null> = {};
      let hdiMap: Record<string, number | null> = {};
      // trendMap[iso3][year] = poverty_rate
      const trendMap: Record<string, Record<number, number>> = {};

      try {
        const [pRes, hRes, tRes] = await Promise.all([
          fetch(povertyUrl),
          fetch(hdiUrl),
          fetch(`https://api.worldbank.org/v2/country/${isos}/indicator/SI.POV.DDAY?date=2014:2023&format=json&per_page=500`),
        ]);
        const [pData, hData, tData] = await Promise.all([
          pRes.json(), hRes.json(), tRes.json(),
        ]);

        if (Array.isArray(pData) && pData[1]) {
          pData[1].forEach((row: { countryiso3code: string; value: number | null }) => {
            povertyMap[row.countryiso3code] = row.value;
          });
        }
        if (Array.isArray(hData) && hData[1]) {
          hData[1].forEach((row: { countryiso3code: string; value: number | null }) => {
            hdiMap[row.countryiso3code] = row.value != null ? row.value / 90 : null;
          });
        }
        if (Array.isArray(tData) && tData[1]) {
          tData[1].forEach((row: { countryiso3code: string; date: string; value: number | null }) => {
            if (row.value != null) {
              if (!trendMap[row.countryiso3code]) trendMap[row.countryiso3code] = {};
              trendMap[row.countryiso3code][parseInt(row.date)] = parseFloat(row.value.toFixed(1));
            }
          });
        }
      } catch {
        // WB API unreachable — features will have null poverty_rate and be filtered out of InsightsFeed
      }

      const features: PovertyFeature[] = SSA_COUNTRIES.map(({ iso3, name, lat, lon }) => {
        const countryHist = trendMap[iso3] ?? {};
        const histYears = Object.keys(countryHist).map(Number).sort();

        // Use most recent available year if the requested year has no data
        const pRate =
          povertyMap[iso3] ??
          (histYears.length > 0 ? countryHist[histYears[histYears.length - 1]] : null);

        // Build ordered poverty rate time series for sparkline
        const povertyTrend = histYears.map((y) => countryHist[y]);

        return {
          country: name,
          iso3,
          lat,
          lon,
          poverty_rate: pRate,
          hdi: hdiMap[iso3] ?? null,
          year,
          ntl_trend: povertyTrend,  // real poverty rate % history (2014–2023, only years with data)
          ndvi_trend: [],
        };
      });

      setPovertyFeatures(features);
    }

    load();
  }, [year, setPovertyFeatures]);
}
