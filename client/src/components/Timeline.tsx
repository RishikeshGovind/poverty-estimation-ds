import { Play, Pause, FastForward } from "lucide-react";
import { useGlobeStore } from "../store/globeStore";
import { useTimeline } from "../hooks/useTimeline";

const MILESTONES: Record<number, string> = {
  2003: "Darfur conflict",
  2011: "East Africa drought",
  2014: "Ebola outbreak",
  2020: "COVID-19 impact",
  2022: "Horn of Africa famine",
};

const YEARS = Array.from({ length: 24 }, (_, i) => 2000 + i);

export default function Timeline() {
  const { setYear, setPlaying } = useGlobeStore();
  const { year, playing } = useTimeline();

  return (
    <footer className="glass fixed bottom-0 left-64 right-72 h-12 z-20 flex items-center px-4 gap-3">
      {/* Play / Pause */}
      <button
        onClick={() => setPlaying(!playing)}
        className="text-cyan-400 hover:text-cyan-300 transition-colors shrink-0"
      >
        {playing ? <Pause size={16} /> : <Play size={16} />}
      </button>

      {/* Year label */}
      <span className="font-mono text-sm text-cyan-400 tabular-nums w-12 shrink-0">{year}</span>

      {/* Timeline track */}
      <div className="relative flex-1 h-6 flex items-center">
        {/* Milestone markers */}
        {YEARS.map((y) => (
          <button
            key={y}
            onClick={() => setYear(y)}
            title={MILESTONES[y]}
            className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 transition-all ${
              MILESTONES[y]
                ? "w-2 h-2 rounded-full bg-amber-400/80 hover:bg-amber-300"
                : "w-1 h-1 rounded-full bg-slate-600 hover:bg-slate-400"
            } ${y === year ? "ring-2 ring-cyan-400 ring-offset-1 ring-offset-black" : ""}`}
            style={{ left: `${((y - 2000) / 23) * 100}%` }}
          />
        ))}

        {/* Slider */}
        <input
          type="range"
          min={2000}
          max={2023}
          step={1}
          value={year}
          onChange={(e) => setYear(parseInt(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer h-full"
        />

        {/* Progress bar */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 h-0.5 bg-cyan-500/40 w-full rounded" />
        <div
          className="absolute left-0 top-1/2 -translate-y-1/2 h-0.5 bg-cyan-400 rounded"
          style={{ width: `${((year - 2000) / 23) * 100}%` }}
        />
      </div>

      {/* Milestone label */}
      <span className="text-[10px] text-amber-400 font-mono w-36 truncate shrink-0">
        {MILESTONES[year] ?? ""}
      </span>

      {/* Speed */}
      <button className="text-slate-500 hover:text-cyan-400 shrink-0">
        <FastForward size={13} />
      </button>

      <span className="text-[10px] font-mono text-slate-600">2000</span>
      <span className="text-[10px] font-mono text-slate-600 ml-auto">2023</span>
    </footer>
  );
}
