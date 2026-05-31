import { useGlobeStore } from "../store/globeStore";
import type { LayerId } from "../store/globeStore";
import {
  Satellite, Leaf, Users, BarChart2, Building,
  AlertTriangle, Droplets, ChevronRight,
} from "lucide-react";

interface LayerDef {
  id: LayerId;
  label: string;
  icon: React.ReactNode;
  color: string;
  description: string;
}

const LAYER_DEFS: LayerDef[] = [
  { id: "nightlights",    label: "Nighttime Lights",    color: "#FBBF24", icon: <Satellite size={14} />,   description: "Electrification proxy" },
  { id: "ndvi",           label: "Vegetation / NDVI",   color: "#22C55E", icon: <Leaf size={14} />,        description: "Food security proxy" },
  { id: "settlements",    label: "Settlement Density",  color: "#F97316", icon: <Users size={14} />,       description: "Urban footprint (Positron)" },
  { id: "poverty",        label: "Poverty Index",       color: "#EF4444", icon: <BarChart2 size={14} />,   description: "World Bank HDI" },
  { id: "infrastructure", label: "Infrastructure",      color: "#60A5FA", icon: <Building size={14} />,   description: "Roads, hospitals, schools (OSM)" },
  { id: "conflict",       label: "Conflict Zones",      color: "#EF4444", icon: <AlertTriangle size={14}/>,description: "ACLED events" },
  { id: "water",          label: "Water Access",        color: "#38BDF8", icon: <Droplets size={14} />,   description: "MODIS water mask (NASA)" },
];

export default function Sidebar() {
  const { layers, toggleLayer, setOpacity } = useGlobeStore();

  return (
    <aside className="glass fixed left-0 top-12 bottom-12 w-64 z-20 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-cyan-500/20">
        <p className="text-[10px] text-cyan-400 font-mono tracking-widest uppercase">Data Layers</p>
      </div>

      {/* Layer list */}
      <div className="flex-1 overflow-y-auto">
        {LAYER_DEFS.map(({ id, label, icon, color, description }) => {
          const active = layers[id].enabled;
          return (
            <div key={id} className="layer-row px-4 py-3 border-b border-white/5">
              <div className="flex items-center gap-2 mb-1">
                {/* Toggle */}
                <button
                  onClick={() => toggleLayer(id)}
                  className={`w-8 h-4 rounded-full transition-all relative ${
                    active ? "bg-cyan-500" : "bg-white/10"
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${
                      active ? "left-4" : "left-0.5"
                    }`}
                  />
                </button>

                {/* Icon + label */}
                <span style={{ color }} className="flex items-center gap-1">
                  {icon}
                  <span className="text-xs font-medium text-slate-200">{label}</span>
                </span>

                <ChevronRight size={10} className="ml-auto text-slate-600" />
              </div>

              <p className="text-[10px] text-slate-500 pl-10">{description}</p>

              {/* Opacity slider — only when active */}
              {active && (
                <div className="flex items-center gap-2 pl-10 mt-2">
                  <span className="text-[10px] text-slate-500 font-mono w-8">
                    {Math.round(layers[id].opacity * 100)}%
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={layers[id].opacity}
                    onChange={(e) => setOpacity(id, parseFloat(e.target.value))}
                    className="flex-1 h-1"
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="px-4 py-3 border-t border-cyan-500/20">
        <p className="text-[10px] text-cyan-400 font-mono tracking-widest uppercase mb-2">Poverty Legend</p>
        <div className="flex gap-1 h-3 rounded overflow-hidden">
          {["#22C55E","#84CC16","#EAB308","#F97316","#EF4444","#991B1B"].map((c, i) => (
            <div key={i} className="flex-1 h-full" style={{ background: c }} />
          ))}
        </div>
        <div className="flex justify-between text-[9px] text-slate-500 font-mono mt-1">
          <span>Low poverty</span>
          <span>Extreme</span>
        </div>
      </div>
    </aside>
  );
}
