// In dev, Vite proxies /api → localhost:8000
// In production (Vercel), set VITE_API_URL to your Render backend URL
const BASE = import.meta.env.VITE_API_URL ?? "";

export const apiUrl = (path: string) => `${BASE}${path}`;
