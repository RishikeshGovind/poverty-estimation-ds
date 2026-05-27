import "./index.css";
import { lazy, Suspense } from "react";
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
import { useModelPredictions } from "./hooks/useModelPredictions";

// Lazy-load Globe so a Cesium init error doesn't prevent the rest of the app from rendering
const Globe = lazy(() => import("./components/Globe"));

function DataLoader() {
  const year = useGlobeStore((s) => s.year);
  useWorldBank(year);
  useACLED(year);
  useModelPredictions(); // overwrites World Bank with real model predictions when available
  return null;
}

export default function App() {
  const { selected, setSelected } = useGlobeStore();

  return (
    <div className="w-screen h-screen overflow-hidden bg-black">
      <DataLoader />

      {/* 3D Globe — lazy loaded so a Cesium error never kills the rest of the UI */}
      <ErrorBoundary fallback={
        <div className="fixed inset-0 bg-[#0b0e1a]" />
      }>
        <Suspense fallback={<div className="fixed inset-0 bg-[#0b0e1a]" />}>
          <Globe onCountryClick={setSelected} />
        </Suspense>
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
