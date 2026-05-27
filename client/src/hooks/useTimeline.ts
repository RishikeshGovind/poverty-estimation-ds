import { useEffect, useRef } from "react";
import { useGlobeStore } from "../store/globeStore";

export function useTimeline() {
  const { playing, year, setYear } = useGlobeStore();
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setYear(
          useGlobeStore.getState().year >= 2023
            ? 2000
            : useGlobeStore.getState().year + 1
        );
      }, 1200);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, setYear]);

  return { year, playing };
}
