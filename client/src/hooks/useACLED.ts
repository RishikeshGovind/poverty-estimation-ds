import { useEffect } from "react";
import { useGlobeStore } from "../store/globeStore";
import type { ConflictEvent } from "../store/globeStore";
import { apiUrl } from "../lib/api";

// Demo conflict data representative of SSA events
// In production: fetch from /api/acled?year=YEAR
const DEMO_EVENTS: ConflictEvent[] = [
  { id:"1", lat: 12.36, lon:  1.52, country:"Burkina Faso", event_type:"Armed Clash",   fatalities:14, date:"2023-08-12", notes:"Attack on civilian convoy near Sahel region" },
  { id:"2", lat:  5.15, lon: 46.20, country:"Somalia",      event_type:"Explosion",      fatalities: 6, date:"2023-09-03", notes:"IED blast near Mogadishu checkpoint" },
  { id:"3", lat: 15.55, lon: 32.53, country:"Sudan",        event_type:"Violence",       fatalities:22, date:"2023-07-18", notes:"RSF-SAF clashes in Khartoum" },
  { id:"4", lat: -1.68, lon: 29.22, country:"DR Congo",     event_type:"Armed Clash",   fatalities: 8, date:"2023-10-01", notes:"M23 engagement in North Kivu" },
  { id:"5", lat: 13.51, lon:  2.12, country:"Niger",        event_type:"Coup",           fatalities: 3, date:"2023-07-26", notes:"Military takeover in Niamey" },
  { id:"6", lat: 17.61, lon:  8.08, country:"Niger",        event_type:"Armed Clash",   fatalities: 5, date:"2023-08-05", notes:"JNIM ambush in Tillabéri" },
  { id:"7", lat:  9.05, lon: 38.74, country:"Ethiopia",     event_type:"Violence",       fatalities:11, date:"2023-09-15", notes:"Amhara regional conflict" },
  { id:"8", lat: 14.10, lon:-15.31, country:"Senegal",      event_type:"Protest",        fatalities: 2, date:"2023-06-03", notes:"Opposition crackdown in Dakar" },
  { id:"9", lat:  6.37, lon:  2.43, country:"Benin",        event_type:"Armed Clash",   fatalities: 4, date:"2023-05-14", notes:"Cross-border incursion from Burkina" },
  { id:"10",lat: 11.56, lon: 43.14, country:"Djibouti",     event_type:"Explosion",      fatalities: 1, date:"2023-04-20", notes:"Device found at port facility" },
  { id:"11",lat:  4.85, lon:31.59,  country:"South Sudan",  event_type:"Armed Clash",   fatalities:18, date:"2023-11-02", notes:"Intercommunal fighting near Jonglei" },
  { id:"12",lat:  8.50, lon:-13.23, country:"Sierra Leone", event_type:"Protest",        fatalities: 3, date:"2023-10-10", notes:"Economic protest turns violent in Freetown" },
];

export function useACLED(year: number) {
  const setConflictEvents = useGlobeStore((s) => s.setConflictEvents);

  useEffect(() => {
    // Set demo data immediately so conflict dots are available before the
    // Render.com backend wakes up (cold starts can take 30-60 s).
    setConflictEvents(DEMO_EVENTS);

    async function load() {
      try {
        const res = await fetch(apiUrl(`/api/acled?year=${year}`));
        if (!res.ok) throw new Error("API unavailable");
        const data = await res.json();
        if (data.events?.length > 0) setConflictEvents(data.events);
      } catch {
        // demo data already set above
      }
    }
    load();
  }, [year, setConflictEvents]);
}
