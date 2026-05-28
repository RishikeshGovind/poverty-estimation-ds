import { useEffect, useState } from "react";
import { X, Satellite, Radio, Zap, Clock, Maximize2 } from "lucide-react";
import { useGlobeStore } from "../store/globeStore";
import { orbitLatLonAlt } from "../utils/orbitPropagator";

export default function SatellitePopup() {
  const selectedSatellite  = useGlobeStore((s) => s.selectedSatellite);
  const satEpochMs         = useGlobeStore((s) => s.satEpochMs);
  const setSelectedSatellite = useGlobeStore((s) => s.setSelectedSatellite);

  const [pos, setPos] = useState<{ lat: number; lon: number; altKm: number } | null>(null);

  useEffect(() => {
    if (!selectedSatellite || !satEpochMs) { setPos(null); return; }

    const refresh = () => {
      const dtSec = (Date.now() - satEpochMs) / 1000;
      setPos(orbitLatLonAlt(selectedSatellite, dtSec, satEpochMs));
    };

    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [selectedSatellite, satEpochMs]);

  if (!selectedSatellite) return null;
  const s = selectedSatellite;

  const typeColor = s.type === "Space Station"
    ? "text-amber-400" : s.type === "Space Telescope"
    ? "text-purple-400" : "text-cyan-400";

  return (
    <div className="glass fixed left-[272px] bottom-14 w-72 z-30 rounded-lg overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-cyan-500/20 bg-cyan-500/5">
        <div className="flex items-center gap-2">
          <Satellite size={14} className="text-cyan-400 shrink-0" />
          <div>
            <h3 className="text-sm font-semibold text-white leading-tight">{s.label}</h3>
            <span className={`text-[10px] font-mono ${typeColor}`}>{s.type}</span>
          </div>
        </div>
        <button onClick={() => setSelectedSatellite(null)} className="text-slate-500 hover:text-white">
          <X size={14} />
        </button>
      </div>

      {/* Agency + mission */}
      <div className="px-4 py-2 border-b border-white/5">
        <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">{s.agency}</p>
        <p className="text-[11px] text-slate-300 mt-0.5">{s.mission}</p>
      </div>

      {/* Orbital parameters grid */}
      <div className="px-4 py-3 grid grid-cols-2 gap-2">
        <div className="bg-white/5 rounded p-2">
          <div className="flex items-center gap-1 mb-0.5">
            <Radio size={9} className="text-slate-500" />
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Altitude</p>
          </div>
          <p className="text-sm font-bold font-mono text-cyan-400">{s.altKm} km</p>
        </div>
        <div className="bg-white/5 rounded p-2">
          <div className="flex items-center gap-1 mb-0.5">
            <Clock size={9} className="text-slate-500" />
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Period</p>
          </div>
          <p className="text-sm font-bold font-mono text-cyan-400">{s.periodMin.toFixed(1)} min</p>
        </div>
        <div className="bg-white/5 rounded p-2">
          <div className="flex items-center gap-1 mb-0.5">
            <Zap size={9} className="text-slate-500" />
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Velocity</p>
          </div>
          <p className="text-sm font-bold font-mono text-amber-400">{s.velocityKms.toFixed(2)} km/s</p>
        </div>
        <div className="bg-white/5 rounded p-2">
          <div className="flex items-center gap-1 mb-0.5">
            <Maximize2 size={9} className="text-slate-500" />
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Inclination</p>
          </div>
          <p className="text-sm font-bold font-mono text-slate-200">{s.incDeg}°</p>
        </div>
      </div>

      {/* Imaging swath */}
      {s.swathKm != null && (
        <div className="px-4 pb-2">
          <div className="flex items-center justify-between bg-green-500/5 border border-green-500/20 rounded p-2">
            <p className="text-[10px] text-green-400 font-mono uppercase tracking-wider">Imaging Swath</p>
            <p className="text-sm font-bold font-mono text-green-400">{s.swathKm} km</p>
          </div>
        </div>
      )}

      {/* Live position — updated every 2 s */}
      <div className="px-4 pb-3">
        <p className="text-[9px] text-cyan-400 font-mono uppercase tracking-widest mb-1.5">
          Live Position {pos ? <span className="text-green-400 animate-pulse">●</span> : null}
        </p>
        <div className="bg-cyan-500/5 border border-cyan-500/20 rounded p-2">
          {pos ? (
            <div className="grid grid-cols-3 gap-1">
              <div>
                <p className="text-[9px] text-slate-500 uppercase">Lat</p>
                <p className="text-[11px] font-mono text-white">{pos.lat.toFixed(1)}°</p>
              </div>
              <div>
                <p className="text-[9px] text-slate-500 uppercase">Lon</p>
                <p className="text-[11px] font-mono text-white">{pos.lon.toFixed(1)}°</p>
              </div>
              <div>
                <p className="text-[9px] text-slate-500 uppercase">Alt</p>
                <p className="text-[11px] font-mono text-white">{pos.altKm.toFixed(0)} km</p>
              </div>
            </div>
          ) : (
            <p className="text-[10px] text-slate-500 text-center">Computing…</p>
          )}
        </div>
        <p className="text-[9px] text-slate-600 mt-1">Orbit groundtrack shown in yellow</p>
      </div>
    </div>
  );
}
