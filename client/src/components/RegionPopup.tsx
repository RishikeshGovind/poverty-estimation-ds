import { X, TrendingUp, TrendingDown, ExternalLink, Zap, Leaf, Building2 } from "lucide-react";
import { useEffect, useState } from "react";
import {
  LineChart, Line, ResponsiveContainer, Tooltip, YAxis,
} from "recharts";
import { useGlobeStore } from "../store/globeStore";
import type { PovertyFeature } from "../store/globeStore";

// ── Sparkline ────────────────────────────────────────────────────────────────
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

// ── Satellite signal row ─────────────────────────────────────────────────────
function SigRow({
  icon, label, value, unit, sub, color,
}: {
  icon: React.ReactNode;
  label: string;
  value: number | undefined;
  unit: string;
  sub?: string;
  color: string;
}) {
  if (value == null) return null;
  return (
    <div className="flex items-center gap-2 py-1.5 border-b border-white/5 last:border-0">
      <span style={{ color }} className="shrink-0">{icon}</span>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] text-slate-400">{label}</p>
        {sub && <p className="text-[9px] text-slate-600 truncate">{sub}</p>}
      </div>
      <span className="text-xs font-mono font-semibold" style={{ color }}>
        {value.toFixed(3)}
        <span className="text-[9px] text-slate-500 ml-0.5">{unit}</span>
      </span>
    </div>
  );
}

// ── Satellite data loader ────────────────────────────────────────────────────
interface SatData {
  ntl_latest?: number;
  ntl_yr_trend?: number;
  ndvi_latest?: number;
  ndbi_latest?: number;
}

function useSatelliteData(f: PovertyFeature): SatData {
  const [sat, setSat] = useState<SatData>({});

  const isCluster = f.iso3.startsWith("CLUSTER|");
  const isCountry = f.iso3.startsWith("COUNTRY|");

  useEffect(() => {
    // DHS cluster features — values embedded directly by phase2_predict.py
    if (isCluster) {
      setSat({
        ntl_latest:   f.ntl_latest,
        ntl_yr_trend: f.ntl_yr_trend,
        ndvi_latest:  f.ndvi_latest,
        ndbi_latest:  f.ndbi_latest,
      });
      return;
    }

    // Country features (geojson synthetic points OR World Bank dots) —
    // fetch from static satellite_features.json (no cold-start risk).
    const iso3 = isCountry
      ? f.iso3.replace("COUNTRY|", "")
      : f.iso3;   // World Bank features have bare ISO3

    if (!iso3 || iso3.length !== 3) return;

    let cancelled = false;
    fetch("/satellite_features.json")
      .then((r) => r.json())
      .then((data: Record<string, { ntl?: Record<string, number>; ndvi?: Record<string, number>; ndbi?: Record<string, number> }>) => {
        if (cancelled) return;
        const entry = data[iso3];
        if (!entry) return;
        const ntl  = entry.ntl  ?? {};
        const ndvi = entry.ndvi ?? {};
        const ndbi = entry.ndbi ?? {};
        const years = Object.keys(ntl).map(Number).sort();
        const ntl_yr_trend = years.length >= 2
          ? (ntl[String(years[years.length - 1])] - ntl[String(years[0])]) / (years[years.length - 1] - years[0])
          : 0;
        setSat({
          ntl_latest:   ntl["2023"] ?? ntl["2022"],
          ntl_yr_trend: ntl_yr_trend,
          ndvi_latest:  ndvi["2023"] ?? ndvi["2022"],
          ndbi_latest:  ndbi["2023"] ?? ndbi["2022"],
        });
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [f.iso3, isCluster, isCountry, f.ntl_latest, f.ntl_yr_trend, f.ndvi_latest, f.ndbi_latest]);

  return sat;
}

// ── Main popup ───────────────────────────────────────────────────────────────
interface Props { feature: PovertyFeature; }

export default function RegionPopup({ feature: f }: Props) {
  const setSelected = useGlobeStore((s) => s.setSelected);
  const sat = useSatelliteData(f);

  const isCluster = f.iso3.startsWith("CLUSTER|");
  const isCountry = f.iso3.startsWith("COUNTRY|");
  const [, adm1Name = "", urbanRural = ""] = isCluster ? f.iso3.split("|") : [];

  const pTrend = f.ntl_trend.length >= 2
    ? f.ntl_trend[f.ntl_trend.length - 1] - f.ntl_trend[0]
    : 0;

  const wealthIndex = (isCluster || isCountry) && f.hdi != null ? (f.hdi * 4 - 2) : null;
  const iso3ForLinks = isCluster || isCountry
    ? f.country.toLowerCase().replace(/\s+/g, "-")
    : f.iso3.toLowerCase();

  const ntlTrend = sat.ntl_yr_trend ?? 0;

  return (
    <div className="glass fixed right-72 bottom-14 w-72 z-30 rounded-lg overflow-hidden shadow-2xl">

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-cyan-500/20">
        <div>
          <h3 className="text-sm font-semibold text-white">{f.country}</h3>
          <span className="text-[10px] text-slate-500 font-mono">
            {isCluster
              ? `${adm1Name || f.country}${urbanRural ? ` · ${urbanRural}` : ""} · DHS 2022`
              : isCountry
              ? `${f.iso3.replace("COUNTRY|", "")} · satellite model`
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
            {isCluster || isCountry ? "Wealth Index" : "HDI Proxy"}
          </p>
          <p className={`text-lg font-bold font-mono ${
            isCluster || isCountry
              ? (wealthIndex ?? 0) >= 0 ? "text-green-400" : "text-red-400"
              : "text-cyan-400"
          }`}>
            {isCluster || isCountry
              ? wealthIndex != null ? wealthIndex.toFixed(2) : "N/A"
              : f.hdi != null ? f.hdi.toFixed(2) : "N/A"}
          </p>
        </div>
      </div>

      {/* Satellite signals ── Phase 4 */}
      {(sat.ntl_latest != null || sat.ndvi_latest != null || sat.ndbi_latest != null) && (
        <div className="px-4 pb-3">
          <p className="text-[10px] text-cyan-400 font-mono uppercase tracking-widest mb-2">
            Satellite Signals · 2023
          </p>
          <div className="bg-white/3 rounded border border-white/8 px-2">
            <SigRow
              icon={<Zap size={11} />}
              label="VIIRS Nighttime Lights"
              value={sat.ntl_latest}
              unit="nW/cm²/sr"
              sub={ntlTrend !== 0
                ? `${ntlTrend > 0 ? "↑" : "↓"} ${Math.abs(ntlTrend).toFixed(4)}/yr — ${ntlTrend > 0.005 ? "rising electrification" : ntlTrend < -0.005 ? "declining" : "stable"}`
                : undefined}
              color="#FBBF24"
            />
            <SigRow
              icon={<Leaf size={11} />}
              label="MODIS Vegetation (NDVI)"
              value={sat.ndvi_latest}
              unit=""
              sub={(sat.ndvi_latest ?? 0) > 0.45 ? "Dense vegetation"
                : (sat.ndvi_latest ?? 0) > 0.3 ? "Moderate vegetation"
                : (sat.ndvi_latest ?? 0) > 0.15 ? "Sparse / semi-arid"
                : "Arid / bare soil"}
              color="#22C55E"
            />
            <SigRow
              icon={<Building2 size={11} />}
              label="Landsat Built-up (NDBI)"
              value={sat.ndbi_latest}
              unit=""
              sub={(sat.ndbi_latest ?? 0) > 0.1 ? "High urban density"
                : (sat.ndbi_latest ?? 0) > 0.03 ? "Moderate built-up"
                : "Low urban / rural"}
              color="#60A5FA"
            />
          </div>
          <p className="text-[9px] text-slate-600 mt-1.5">
            Sources: NASA VIIRS · MODIS MOD13A3 · Landsat 8/9 via GEE
          </p>
        </div>
      )}

      {/* WB poverty trend sparkline */}
      {!isCluster && !isCountry && f.ntl_trend.length >= 2 && (
        <div className="px-4 pb-2">
          <div className="flex items-center justify-between mb-1">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Poverty Trend (%)</p>
            <span className={`flex items-center gap-0.5 text-[10px] font-mono ${
              pTrend < 0 ? "text-green-400" : "text-red-400"
            }`}>
              {pTrend < 0 ? <TrendingDown size={10} /> : <TrendingUp size={10} />}
              {pTrend < 0 ? "" : "+"}{pTrend.toFixed(1)}pp
            </span>
          </div>
          <Sparkline data={f.ntl_trend} color="#FBBF24" />
          <p className="text-[9px] text-slate-600 mt-0.5">Source: World Bank SI.POV.DDAY</p>
        </div>
      )}

      {/* Model info for DHS clusters */}
      {isCluster && (
        <div className="px-4 pb-3">
          <div className="bg-cyan-500/5 border border-cyan-500/20 rounded p-2">
            <p className="text-[10px] text-cyan-400 font-mono uppercase tracking-wider mb-1">
              Model Prediction · R²=0.776
            </p>
            <p className="text-[10px] text-slate-400 leading-relaxed">
              Predicted from VIIRS nighttime lights + Sentinel-2 visible bands.
              Trained on 3,048 DHS 2022–23 clusters across Kenya & Nigeria.
            </p>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="px-4 pb-3 flex gap-2">
        <a
          href={`https://data.worldbank.org/country/${iso3ForLinks}`}
          target="_blank" rel="noreferrer"
          className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 rounded px-3 py-1.5"
        >
          <ExternalLink size={10} /> World Bank
        </a>
        <a
          href={`https://hdr.undp.org/data-center/specific-country-data#/countries/${iso3ForLinks}`}
          target="_blank" rel="noreferrer"
          className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 rounded px-3 py-1.5"
        >
          <ExternalLink size={10} /> UNDP HDR
        </a>
      </div>
    </div>
  );
}
