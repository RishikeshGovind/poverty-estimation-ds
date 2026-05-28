import { TrendingUp, TrendingDown, Minus, MapPin } from "lucide-react";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

// delta = poverty rate change (pp): negative means improvement, positive means worsening
function TrendBadge({ delta }: { delta: number | null }) {
  if (delta === null) return null;
  // Poverty falling = good (green), rising = bad (red)
  if (delta < -2) return <span className="flex items-center gap-0.5 text-green-400 text-[10px]"><TrendingDown size={10}/>{Math.abs(delta).toFixed(1)}pp</span>;
  if (delta > 2)  return <span className="flex items-center gap-0.5 text-red-400   text-[10px]"><TrendingUp   size={10}/>+{delta.toFixed(1)}pp</span>;
  return                 <span className="flex items-center gap-0.5 text-slate-500  text-[10px]"><Minus        size={10}/>stable</span>;
}

function InsightCard({ f, rank }: { f: PovertyFeature; rank: number }) {
  const { setSelected, setFlyTo } = useGlobeStore();
  const pRate = f.poverty_rate ?? 50;
  const isHigh = pRate > 55;

  // Real delta from World Bank historical series (first→last known value)
  const trend = f.ntl_trend;
  const delta = trend.length >= 2 ? trend[trend.length - 1] - trend[0] : null;

  return (
    <div className="border-b border-white/5 px-3 py-3 hover:bg-white/3 transition-colors cursor-default">
      <div className="flex items-start gap-2">
        {/* Rank badge */}
        <span className={`shrink-0 w-5 h-5 rounded text-[9px] font-mono flex items-center justify-center ${
          rank <= 3 ? "bg-red-500/20 text-red-400" : "bg-white/5 text-slate-500"
        }`}>
          {rank}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-200 truncate">{f.country}</span>
            <TrendBadge delta={delta} />
          </div>

          {/* Poverty bar */}
          <div className="mt-1 h-1 bg-white/10 rounded overflow-hidden">
            <div
              className={`h-full rounded transition-all ${isHigh ? "bg-red-500" : "bg-amber-400"}`}
              style={{ width: `${Math.min(pRate, 100)}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
            {pRate.toFixed(1)}% below poverty line
          </p>

          {/* Auto-insight */}
          <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
            {isHigh
              ? `${f.country} shows ${pRate.toFixed(0)}% extreme poverty — NTL intensity remains critically low.`
              : `Moderate poverty in ${f.country}; NTL trend improving since ${f.year - 3}.`}
          </p>
        </div>
      </div>

      {/* Fly-to button */}
      <button
        onClick={() => { setFlyTo([f.lat, f.lon, 1_200_000]); setSelected(f); }}
        className="mt-2 ml-7 flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300"
      >
        <MapPin size={9} /> Fly To
      </button>
    </div>
  );
}

export default function InsightsFeed() {
  const { povertyFeatures } = useGlobeStore();

  // Aggregate clusters by country — many DHS clusters → one card per country
  const countryMap = new Map<string, { sum: number; count: number; rep: PovertyFeature }>();
  povertyFeatures.filter((f) => f.poverty_rate != null).forEach((f) => {
    if (!countryMap.has(f.country)) {
      countryMap.set(f.country, { sum: 0, count: 0, rep: f });
    }
    const e = countryMap.get(f.country)!;
    e.sum += f.poverty_rate ?? 0;
    e.count++;
  });
  const aggregated: PovertyFeature[] = Array.from(countryMap.values()).map(({ sum, count, rep }) => ({
    ...rep,
    poverty_rate: sum / count,
  }));

  const sorted = aggregated
    .sort((a, b) => (b.poverty_rate ?? 0) - (a.poverty_rate ?? 0))
    .slice(0, 15);

  return (
    <aside className="glass fixed right-0 top-12 bottom-12 w-72 z-20 flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-cyan-500/20">
        <p className="text-[10px] text-cyan-400 font-mono tracking-widest uppercase">
          Insights Feed
        </p>
        <p className="text-[9px] text-slate-500 mt-0.5">Ranked by poverty severity</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {sorted.length === 0 ? (
          <div className="p-4 text-xs text-slate-500 text-center">Loading data…</div>
        ) : (
          sorted.map((f, i) => <InsightCard key={f.country} f={f} rank={i + 1} />)
        )}
      </div>

      {/* Summary footer */}
      {sorted.length > 0 && (
        <div className="px-4 py-3 border-t border-cyan-500/20 grid grid-cols-2 gap-2">
          <div className="bg-white/5 rounded p-2">
            <p className="text-[9px] text-slate-500 uppercase">Countries tracked</p>
            <p className="text-sm font-mono font-bold text-cyan-400">
              {new Set(povertyFeatures.map((f) => f.country)).size}
            </p>
          </div>
          <div className="bg-white/5 rounded p-2">
            <p className="text-[9px] text-slate-500 uppercase">Avg poverty</p>
            <p className="text-sm font-mono font-bold text-red-400">
              {(
                sorted.reduce((s, f) => s + (f.poverty_rate ?? 0), 0) /
                Math.max(sorted.length, 1)
              ).toFixed(1)}%
            </p>
          </div>
        </div>
      )}
    </aside>
  );
}
