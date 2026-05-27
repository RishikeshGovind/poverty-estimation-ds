import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";

const root = document.getElementById("root")!;

try {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>
  );
} catch (e) {
  root.innerHTML = `<div style="color:#ef4444;font-family:monospace;padding:2rem;background:#000;height:100vh">
    <b>Fatal startup error</b><br/><br/>${e instanceof Error ? e.message : String(e)}
  </div>`;
}
