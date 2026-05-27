import "./index.css";
import { Ion } from "cesium";
import Globe from "./components/Globe";
import TopBar from "./components/TopBar";
import Sidebar from "./components/Sidebar";
import Timeline from "./components/Timeline";
import InsightsFeed from "./components/InsightsFeed";
import AskAfricaLens from "./components/AskAfricaLens";
import RegionPopup from "./components/RegionPopup";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useGlobeStore } from "./store/globeStore";
import { useWorldBank } from "./hooks/useWorldBank";
import { useACLED } from "./hooks/useACLED";

// Set Cesium Ion token from env
Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN ?? "";

function DataLoader() {
  const year = useGlobeStore((s) => s.year);
  useWorldBank(year);
  useACLED(year);
  return null;
}

export default function App() {
  const { selected, setSelected } = useGlobeStore();

  return (
    <div className="w-screen h-screen overflow-hidden bg-black">
      <DataLoader />

      {/* 3D Globe — fills entire screen */}
      <ErrorBoundary fallback={
        <div className="fixed inset-0 bg-black flex items-center justify-center">
          <p className="text-slate-500 text-xs font-mono">Globe failed to load — check console</p>
        </div>
      }>
        <Globe onCountryClick={setSelected} />
      </ErrorBoundary>

      {/* UI chrome on top */}
      <TopBar />
      <Sidebar />
      <InsightsFeed />
      <Timeline />
      <AskAfricaLens />

      {/* Country popup on click */}
      {selected && <RegionPopup feature={selected} />}
    </div>
  );
}
