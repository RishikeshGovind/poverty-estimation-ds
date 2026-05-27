import { useState, useRef, useEffect } from "react";
import { Send, Bot, ChevronDown, ChevronUp } from "lucide-react";
import { useGlobeStore } from "../store/globeStore";
import { apiUrl } from "../lib/api";

export default function AskAfricaLens() {
  const [open, setOpen] = useState(false);
  const { aiQuery, setAiQuery, aiResponse, setAiResponse, aiLoading, setAiLoading, povertyFeatures, setFlyTo } = useGlobeStore();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!aiQuery.trim() || aiLoading) return;
    setAiLoading(true);
    setAiResponse("");

    try {
      const res = await fetch(apiUrl("/api/ai-query"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: aiQuery,
          context: povertyFeatures.map((f) => ({
            country: f.country, iso3: f.iso3, poverty_rate: f.poverty_rate, hdi: f.hdi,
          })),
        }),
      });

      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setAiResponse(data.answer ?? "No response.");

      // If API returns a country to fly to
      if (data.fly_to) {
        const match = povertyFeatures.find(
          (f) => f.iso3 === data.fly_to || f.country.toLowerCase() === data.fly_to.toLowerCase()
        );
        if (match) setFlyTo([match.lat, match.lon, 1_500_000]);
      }
    } catch {
      // Offline: generate a local summary
      const sorted = [...povertyFeatures]
        .filter((f) => f.poverty_rate != null)
        .sort((a, b) => (b.poverty_rate ?? 0) - (a.poverty_rate ?? 0));

      setAiResponse(
        `Based on World Bank data, the most critically affected region is **${sorted[0]?.country ?? "N/A"}** ` +
        `(${sorted[0]?.poverty_rate?.toFixed(1) ?? "?"}% poverty rate). ` +
        `Top 3 by poverty severity: ${sorted.slice(0, 3).map((f) => f.country).join(", ")}.`
      );
    }

    setAiLoading(false);
    setAiQuery("");
  }

  return (
    <div className="glass fixed bottom-12 left-64 right-72 z-25">
      {/* Toggle bar */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-4 py-2 border-t border-cyan-500/20 hover:bg-cyan-500/5 transition-colors"
      >
        <Bot size={13} className="text-cyan-400" />
        <span className="text-[11px] text-slate-400 font-mono">Ask AfricaLens…</span>
        {open ? <ChevronDown size={12} className="ml-auto text-slate-600" /> : <ChevronUp size={12} className="ml-auto text-slate-600" />}
      </button>

      {/* Expanded panel */}
      {open && (
        <div className="px-4 pb-3 pt-1 border-t border-white/5">
          {/* Response */}
          {(aiResponse || aiLoading) && (
            <div className="mb-2 bg-white/5 rounded p-3 text-xs text-slate-300 leading-relaxed max-h-24 overflow-y-auto">
              {aiLoading ? (
                <span className="text-cyan-400 animate-pulse">Analysing…</span>
              ) : (
                aiResponse
              )}
            </div>
          )}

          {/* Prompt suggestions */}
          {!aiResponse && !aiLoading && (
            <div className="flex flex-wrap gap-1 mb-2">
              {[
                "Which countries deteriorated most since 2015?",
                "Where is poverty most extreme right now?",
                "Show me regions with improving NTL trends",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => setAiQuery(q)}
                  className="text-[9px] border border-cyan-500/25 text-slate-500 hover:text-cyan-400 hover:border-cyan-400/50 rounded px-2 py-0.5 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={aiQuery}
              onChange={(e) => setAiQuery(e.target.value)}
              placeholder="Ask a question about poverty in Africa…"
              className="flex-1 bg-white/5 border border-white/10 focus:border-cyan-500/60 rounded text-xs px-3 py-1.5 text-slate-200 outline-none"
            />
            <button
              type="submit"
              disabled={aiLoading || !aiQuery.trim()}
              className="px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 rounded text-cyan-400 disabled:opacity-40 transition-all"
            >
              <Send size={12} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
