import { create } from "zustand";
import type { SatelliteInfo } from "../utils/orbitPropagator";

export type LayerId =
  | "nightlights"
  | "ndvi"
  | "settlements"
  | "poverty"
  | "infrastructure"
  | "conflict"
  | "water";

export interface LayerState {
  enabled: boolean;
  opacity: number;
}

export interface ConflictEvent {
  id: string;
  lat: number;
  lon: number;
  country: string;
  event_type: string;
  fatalities: number;
  date: string;
  notes: string;
}

export interface PovertyFeature {
  country: string;
  iso3: string;
  lat: number;
  lon: number;
  poverty_rate: number | null;
  hdi: number | null;
  year: number;
  ntl_trend: number[];
  ndvi_trend: number[];
  // Real satellite signals (Phase 4)
  ntl_latest?: number;   // VIIRS NTL radiance nW/cm²/sr
  ntl_yr_trend?: number; // slope per year
  ndvi_latest?: number;  // MODIS NDVI [0–1]
  ndbi_latest?: number;  // Landsat NDBI [−1–1]
}

interface GlobeStore {
  // Active year on the timeline
  year: number;
  setYear: (y: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;

  // Per-layer state
  layers: Record<LayerId, LayerState>;
  toggleLayer: (id: LayerId) => void;
  setOpacity: (id: LayerId, v: number) => void;

  // Selected country popup
  selected: PovertyFeature | null;
  setSelected: (f: PovertyFeature | null) => void;

  // Conflict events loaded from ACLED
  conflictEvents: ConflictEvent[];
  setConflictEvents: (ev: ConflictEvent[]) => void;

  // Poverty features loaded from World Bank
  povertyFeatures: PovertyFeature[];
  setPovertyFeatures: (f: PovertyFeature[]) => void;

  // AI chat
  aiQuery: string;
  setAiQuery: (q: string) => void;
  aiResponse: string;
  setAiResponse: (r: string) => void;
  aiLoading: boolean;
  setAiLoading: (v: boolean) => void;

  // Fly-to target (lat, lon, height)
  flyTo: [number, number, number] | null;
  setFlyTo: (t: [number, number, number] | null) => void;

  // Selected satellite popup
  selectedSatellite: SatelliteInfo | null;
  setSelectedSatellite: (s: SatelliteInfo | null) => void;
  satEpochMs: number;
  setSatEpochMs: (ms: number) => void;

  // Selected conflict event popup
  selectedConflict: ConflictEvent | null;
  setSelectedConflict: (e: ConflictEvent | null) => void;
}

const DEFAULT_LAYERS: Record<LayerId, LayerState> = {
  nightlights:    { enabled: false, opacity: 0.80 },
  ndvi:           { enabled: false, opacity: 0.75 },
  settlements:    { enabled: false, opacity: 0.70 },
  poverty:        { enabled: true,  opacity: 0.80 },
  infrastructure: { enabled: false, opacity: 0.90 },
  conflict:       { enabled: false, opacity: 1.00 },
  water:          { enabled: false, opacity: 0.70 },
};

export const useGlobeStore = create<GlobeStore>((set) => ({
  year: 2023,
  setYear: (year) => set({ year }),
  playing: false,
  setPlaying: (playing) => set({ playing }),

  layers: DEFAULT_LAYERS,
  toggleLayer: (id) =>
    set((s) => ({
      layers: {
        ...s.layers,
        [id]: { ...s.layers[id], enabled: !s.layers[id].enabled },
      },
    })),
  setOpacity: (id, v) =>
    set((s) => ({
      layers: { ...s.layers, [id]: { ...s.layers[id], opacity: v } },
    })),

  selected: null,
  setSelected: (selected) => set({ selected }),

  conflictEvents: [],
  setConflictEvents: (conflictEvents) => set({ conflictEvents }),

  povertyFeatures: [],
  setPovertyFeatures: (povertyFeatures) => set({ povertyFeatures }),

  aiQuery: "",
  setAiQuery: (aiQuery) => set({ aiQuery }),
  aiResponse: "",
  setAiResponse: (aiResponse) => set({ aiResponse }),
  aiLoading: false,
  setAiLoading: (aiLoading) => set({ aiLoading }),

  flyTo: null,
  setFlyTo: (flyTo) => set({ flyTo }),

  selectedSatellite: null,
  setSelectedSatellite: (selectedSatellite) => set({ selectedSatellite }),
  satEpochMs: 0,
  setSatEpochMs: (satEpochMs) => set({ satEpochMs }),

  selectedConflict: null,
  setSelectedConflict: (selectedConflict) => set({ selectedConflict }),
}));
