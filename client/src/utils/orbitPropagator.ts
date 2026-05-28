// Keplerian orbit propagator — shared between Globe.tsx and SatellitePopup.tsx
export const MU      = 3.986e14;      // Earth gravitational parameter m³/s²
export const R_EARTH = 6.371e6;       // Earth mean radius m
const OMEGA_E        = 7.2921150e-5;  // Earth rotation rate rad/s
const J2000_S        = 946728000;     // Unix timestamp of J2000 epoch s

export interface Orbit {
  label: string;
  altKm: number;
  incDeg: number;
  raanDeg: number;
  nu0Deg: number;
}

export interface SatelliteInfo extends Orbit {
  type: string;
  agency: string;
  mission: string;
  swathKm: number | null;
  periodMin: number;
  velocityKms: number;
}

const META: Record<string, { type: string; agency: string; mission: string; swathKm: number | null }> = {
  "ISS":          { type: "Space Station",    agency: "NASA / Roscosmos / ESA / JAXA",  mission: "Crewed microgravity research",              swathKm: null },
  "Tiangong":     { type: "Space Station",    agency: "CNSA",                           mission: "Crewed modular station",                    swathKm: null },
  "Sentinel-2A":  { type: "Earth Observation",agency: "ESA / Copernicus",               mission: "Multispectral land imaging",                swathKm: 290  },
  "Sentinel-2B":  { type: "Earth Observation",agency: "ESA / Copernicus",               mission: "Multispectral land imaging",                swathKm: 290  },
  "Landsat-8":    { type: "Earth Observation",agency: "USGS / NASA",                    mission: "Multispectral land-surface imaging",        swathKm: 185  },
  "Landsat-9":    { type: "Earth Observation",agency: "USGS / NASA",                    mission: "Multispectral land-surface imaging",        swathKm: 185  },
  "VIIRS / SNPP": { type: "Earth Observation",agency: "NASA / NOAA",                    mission: "Nighttime lights · weather imaging",        swathKm: 3040 },
  "MODIS Terra":  { type: "Earth Observation",agency: "NASA",                           mission: "Daily land / ocean / atmosphere composite", swathKm: 2330 },
  "MODIS Aqua":   { type: "Earth Observation",agency: "NASA",                           mission: "Daily ocean / atmosphere composite",        swathKm: 2330 },
  "Hubble":       { type: "Space Telescope",  agency: "NASA / ESA",                     mission: "Deep-space optical / UV / IR imaging",      swathKm: null },
};

const BASE_ORBITS: Orbit[] = [
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

export const SAT_ORBITS: SatelliteInfo[] = BASE_ORBITS.map((o) => {
  const a = R_EARTH + o.altKm * 1e3;
  const T = 2 * Math.PI * Math.sqrt((a * a * a) / MU);
  const v = Math.sqrt(MU / a);
  const m = META[o.label] ?? { type: "Unknown", agency: "Unknown", mission: "Unknown", swathKm: null };
  return { ...o, ...m, periodMin: T / 60, velocityKms: v / 1000 };
});

/** Returns ECI→ECEF Cartesian position as [x, y, z] metres. */
export function orbitXYZ(orbit: Orbit, dtSec: number, epochMs: number): [number, number, number] {
  const a    = R_EARTH + orbit.altKm * 1e3;
  const n    = Math.sqrt(MU / (a * a * a));
  const inc  = orbit.incDeg  * Math.PI / 180;
  const raan = orbit.raanDeg * Math.PI / 180;
  const nu   = (orbit.nu0Deg * Math.PI / 180) + n * dtSec;

  const xP = a * Math.cos(nu), yP = a * Math.sin(nu);
  const xE = xP * Math.cos(raan) - yP * Math.cos(inc) * Math.sin(raan);
  const yE = xP * Math.sin(raan) + yP * Math.cos(inc) * Math.cos(raan);
  const zE = yP * Math.sin(inc);

  const t0   = epochMs / 1000 - J2000_S;
  const gmst = (280.46061837 + 360.98564736629 * t0 / 86400) * Math.PI / 180
             + OMEGA_E * dtSec;

  return [
    xE * Math.cos(gmst) + yE * Math.sin(gmst),
   -xE * Math.sin(gmst) + yE * Math.cos(gmst),
    zE,
  ];
}

/** Returns geodetic lat (°), lon (°), altitude (km) for the given time offset. */
export function orbitLatLonAlt(
  orbit: Orbit,
  dtSec: number,
  epochMs: number,
): { lat: number; lon: number; altKm: number } {
  const [x, y, z] = orbitXYZ(orbit, dtSec, epochMs);
  const r   = Math.sqrt(x * x + y * y + z * z);
  const lat = Math.asin(z / r) * (180 / Math.PI);
  const lon = Math.atan2(y, x) * (180 / Math.PI);
  return { lat, lon, altKm: (r - R_EARTH) / 1e3 };
}
