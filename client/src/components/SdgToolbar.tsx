import { useGlobeStore } from "../store/globeStore";
import type { SdgLayer } from "../store/globeStore";

const TABS: { id: SdgLayer; label: string; color: string }[] = [
  { id: "composite", label: "Composite",        color: "#6366f1" },
  { id: "sdg1",      label: "SDG 1 · Poverty",  color: "#EF4444" },
  { id: "sdg7",      label: "SDG 7 · Energy",   color: "#FBBF24" },
  { id: "sdg11",     label: "SDG 11 · Cities",  color: "#60A5FA" },
];

export default function SdgToolbar() {
  const { sdgLayer, setSdgLayer } = useGlobeStore();

  return (
    <div className="fixed top-16 left-1/2 -translate-x-1/2 z-20 flex gap-1 glass rounded-full px-2.5 py-1.5 border border-cyan-500/20">
      {TABS.map(({ id, label, color }) => {
        const active = sdgLayer === id;
        return (
          <button
            key={id}
            onClick={() => setSdgLayer(id)}
            className={`px-3 py-1 rounded-full text-[11px] font-semibold transition-all whitespace-nowrap ${
              active ? "text-white" : "text-slate-400 hover:text-white"
            }`}
            style={active ? { background: color } : undefined}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
