import "./index.css";
import { Ion } from "cesium";
import Globe from "./components/Globe";
import TopBar from "./components/TopBar";
import Sidebar from "./components/Sidebar";
import Timeline from "./components/Timeline";
import InsightsFeed from "./components/InsightsFeed";
import AskAfricaLens from "./components/AskAfricaLens";
import RegionPopup from "./components/RegionPopup";
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
      <Globe onCountryClick={setSelected} />

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
