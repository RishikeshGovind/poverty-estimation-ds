import { X, TrendingUp, TrendingDown, ExternalLink } from "lucide-react";
import {
  LineChart, Line, ResponsiveContainer, Tooltip, YAxis,
} from "recharts";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const chartData = data.map((v, i) => ({ i, v }));
  return (
    <ResponsiveContainer width="100%" height={40}>
      <LineChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <YAxis domain={["auto", "auto"]} hide />
        <Tooltip
          contentStyle={{ background: "#0a0a14", border: "1px solid #00FFFF33", fontSize: 10 }}
          labelFormatter={() => ""}
          formatter={(v) => [typeof v === "number" ? `${v.toFixed(1)}%` : v, "Poverty rate"]}
        />
        <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

interface Props {
  feature: PovertyFeature;
}

export default function RegionPopup({ feature: f }: Props) {
  const setSelected = useGlobeStore((s) => s.setSelected);

  // Cluster features encode place info as "CLUSTER|adm1|urban"
  const isCluster = f.iso3.startsWith("CLUSTER|");
  const [, adm1Name = "", urbanRural = ""] = isCluster ? f.iso3.split("|") : [];
  // Real poverty rate change (percentage points) from WB historical data
  const pTrend = f.ntl_trend.length >= 2
    ? f.ntl_trend[f.ntl_trend.length - 1] - f.ntl_trend[0]
    : 0;

  const wealthIndex = isCluster && f.hdi != null ? (f.hdi * 4 - 2) : null;
  const countrySlug = f.country.toLowerCase().replace(/\s+/g, "-");

  return (
    <div className="glass fixed right-72 bottom-14 w-72 z-30 rounded-lg overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-cyan-500/20">
        <div>
          <h3 className="text-sm font-semibold text-white">{f.country}</h3>
          <span className="text-[10px] text-slate-500 font-mono">
            {isCluster
              ? `${adm1Name || f.country}${urbanRural ? ` · ${urbanRural}` : ""} · DHS 2022`
              : `${f.iso3} · ${f.year}`}
          </span>
        </div>
        <button onClick={() => setSelected(null)} className="text-slate-500 hover:text-white">
          <X size={14} />
        </button>
      </div>

      {/* Key metrics */}
      <div className="px-4 py-3 grid grid-cols-2 gap-3">
        <div className="bg-white/5 rounded p-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Poverty Rate</p>
          <p className="text-lg font-bold font-mono" style={{
            color: (f.poverty_rate ?? 50) > 50 ? "#EF4444" : "#F59E0B",
          }}>
            {f.poverty_rate != null ? `${f.poverty_rate.toFixed(1)}%` : "N/A"}
          </p>
        </div>
        <div className="bg-white/5 rounded p-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">
            {isCluster ? "Wealth Index" : "HDI Proxy"}
          </p>
          <p className={`text-lg font-bold font-mono ${
            isCluster
              ? (wealthIndex ?? 0) >= 0 ? "text-green-400" : "text-red-400"
              : "text-cyan-400"
          }`}>
            {isCluster
              ? wealthIndex != null ? wealthIndex.toFixed(2) : "N/A"
              : f.hdi != null ? f.hdi.toFixed(2) : "N/A"}
          </p>
        </div>
      </div>

      {/* Poverty rate trend — WB features only, requires ≥2 data points */}
      {!isCluster && f.ntl_trend.length >= 2 && (
        <div className="px-4 pb-2">
          <div className="flex items-center justify-between mb-1">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Poverty Rate Trend (%)</p>
            <span className={`flex items-center gap-0.5 text-[10px] font-mono ${
              pTrend < 0 ? "text-green-400" : "text-red-400"
            }`}>
              {pTrend < 0 ? <TrendingDown size={10} /> : <TrendingUp size={10} />}
              {pTrend < 0 ? "" : "+"}{(pTrend).toFixed(1)}pp
            </span>
          </div>
          <Sparkline data={f.ntl_trend} color="#FBBF24" />
          <p className="text-[9px] text-slate-600 mt-0.5">Source: World Bank SI.POV.DDAY, 2014–2023</p>
        </div>
      )}

      {/* Cluster model info */}
      {isCluster && (
        <div className="px-4 pb-3">
          <div className="bg-cyan-500/5 border border-cyan-500/20 rounded p-2">
            <p className="text-[10px] text-cyan-400 font-mono uppercase tracking-wider mb-1">
              Model Prediction
            </p>
            <p className="text-[10px] text-slate-400 leading-relaxed">
              Predicted from Sentinel-2 + VIIRS nighttime lights.
              R² = 0.70 across Kenya & Nigeria (2022–2023 DHS).
            </p>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="px-4 pb-3 flex gap-2">
        <a
          href={`https://data.worldbank.org/country/${isCluster ? countrySlug : f.iso3.toLowerCase()}`}
          target="_blank" rel="noreferrer"
          className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 rounded px-3 py-1.5"
        >
          <ExternalLink size={10} /> World Bank
        </a>
        <a
          href={`https://hdr.undp.org/data-center/specific-country-data#/countries/${isCluster ? countrySlug : f.iso3}`}
          target="_blank" rel="noreferrer"
          className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 rounded px-3 py-1.5"
        >
          <ExternalLink size={10} /> UNDP HDR
        </a>
      </div>
    </div>
  );
}
