import { X, AlertTriangle, ExternalLink } from "lucide-react";
import { useGlobeStore } from "../store/globeStore";

const TYPE_COLOR: Record<string, string> = {
  "Armed Clash":  "#EF4444",
  "Explosion":    "#F97316",
  "Violence":     "#EF4444",
  "Coup":         "#A855F7",
  "Protest":      "#F59E0B",
};

export default function ConflictPopup() {
  const ev              = useGlobeStore((s) => s.selectedConflict)!;
  const setSelectedConflict = useGlobeStore((s) => s.setSelectedConflict);

  const typeColor = TYPE_COLOR[ev.event_type] ?? "#EF4444";

  return (
    <div className="glass fixed left-72 bottom-14 w-72 z-30 rounded-lg overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-red-500/30"
           style={{ borderColor: `${typeColor}33` }}>
        <div className="flex items-center gap-2">
          <AlertTriangle size={14} style={{ color: typeColor }} />
          <div>
            <h3 className="text-sm font-semibold text-white">{ev.country}</h3>
            <span className="text-[10px] text-slate-500 font-mono">{ev.date}</span>
          </div>
        </div>
        <button onClick={() => setSelectedConflict(null)} className="text-slate-500 hover:text-white">
          <X size={14} />
        </button>
      </div>

      {/* Event type + fatalities */}
      <div className="px-4 py-3 grid grid-cols-2 gap-3">
        <div className="bg-white/5 rounded p-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Event Type</p>
          <p className="text-xs font-semibold" style={{ color: typeColor }}>{ev.event_type}</p>
        </div>
        <div className="bg-white/5 rounded p-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Fatalities</p>
          <p className="text-lg font-bold font-mono" style={{ color: typeColor }}>
            {ev.fatalities}
          </p>
        </div>
      </div>

      {/* Notes */}
      {ev.notes && (
        <div className="px-4 pb-3">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Details</p>
          <p className="text-[11px] text-slate-300 leading-relaxed">{ev.notes}</p>
        </div>
      )}

      {/* Coordinates + ACLED link */}
      <div className="px-4 pb-3 flex items-center justify-between">
        <span className="text-[9px] text-slate-600 font-mono">
          {ev.lat.toFixed(2)}°, {ev.lon.toFixed(2)}°
        </span>
        <a
          href="https://acleddata.com/data-export-tool/"
          target="_blank" rel="noreferrer"
          className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 rounded px-3 py-1.5"
        >
          <ExternalLink size={10} /> ACLED Data
        </a>
      </div>
    </div>
  );
}
