import { useState, useEffect } from "react";
import { Search, Globe2, MapPin } from "lucide-react";
import { useGlobeStore } from "../store/globeStore";

const REGIONS: Record<string, [number, number, number]> = {
  "East Africa":     [  0.0,  37.0, 4_000_000],
  "West Africa":     [ 10.0,  -5.0, 4_000_000],
  "North Africa":    [ 26.0,  20.0, 5_000_000],
  "Central Africa":  [ -2.0,  22.0, 4_000_000],
  "Southern Africa": [-25.0,  25.0, 4_000_000],
};

function UTCClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span className="font-mono text-xs text-cyan-400 tabular-nums">
      {time.toUTCString().slice(17, 25)} UTC
    </span>
  );
}

export default function TopBar() {
  const { setFlyTo, povertyFeatures } = useGlobeStore();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<typeof povertyFeatures>([]);

  function handleSearch(q: string) {
    setQuery(q);
    if (!q.trim()) { setResults([]); return; }
    setResults(
      povertyFeatures
        .filter((f) => f.country.toLowerCase().includes(q.toLowerCase()))
        .slice(0, 5)
    );
  }

  return (
    <header className="glass fixed top-0 left-0 right-0 h-12 z-30 flex items-center px-4 gap-6">
      {/* Logo */}
      <div className="flex items-center gap-2 shrink-0">
        <Globe2 size={18} className="text-cyan-400" />
        <span className="text-sm font-bold text-white tracking-wide">AfricaLens</span>
        <span className="hidden sm:block text-[10px] text-slate-500 font-mono ml-1">
          SATELLITE INTELLIGENCE FOR HUMAN DEVELOPMENT
        </span>
      </div>

      {/* Region quick-select */}
      <nav className="hidden md:flex gap-1">
        {Object.entries(REGIONS).map(([label, coords]) => (
          <button
            key={label}
            onClick={() => setFlyTo(coords)}
            className="text-[10px] text-slate-400 hover:text-cyan-400 px-2 py-1 rounded border border-transparent hover:border-cyan-500/40 transition-all font-mono uppercase tracking-wider"
          >
            {label.replace(" Africa", "")}
          </button>
        ))}
      </nav>

      {/* Search */}
      <div className="ml-auto relative w-52">
        <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          placeholder="Search country…"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          className="w-full bg-white/5 border border-white/10 focus:border-cyan-500/60 rounded text-xs pl-7 pr-3 py-1.5 text-slate-200 outline-none"
        />
        {results.length > 0 && (
          <ul className="absolute top-8 right-0 w-full glass rounded shadow-lg overflow-hidden z-50">
            {results.map((f) => (
              <li key={f.iso3}>
                <button
                  onClick={() => { setFlyTo([f.lat, f.lon, 1_500_000]); setQuery(""); setResults([]); }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-xs text-slate-300 hover:bg-cyan-500/10 text-left"
                >
                  <MapPin size={10} className="text-cyan-400 shrink-0" />
                  {f.country}
                  {f.poverty_rate != null && (
                    <span className="ml-auto text-red-400 font-mono">{f.poverty_rate.toFixed(1)}%</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* UTC Clock */}
      <div className="shrink-0 pl-2 border-l border-white/10">
        <UTCClock />
      </div>
    </header>
  );
}
