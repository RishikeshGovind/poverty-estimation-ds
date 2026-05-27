import { Component } from "react";
import type { ReactNode } from "react";

interface Props { children: ReactNode; fallback?: ReactNode; }
interface State { error: Error | null; }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      return this.props.fallback ?? (
        <div className="fixed inset-0 flex items-center justify-center bg-black">
          <div className="glass rounded-lg p-6 max-w-md text-center">
            <p className="text-red-400 text-sm font-mono mb-2">Component error</p>
            <p className="text-slate-400 text-xs">{this.state.error.message}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
