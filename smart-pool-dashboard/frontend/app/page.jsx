"use client";
import { useSocket } from "./useSocket";
import {
  CrowdCard, DrowningCard, GarbageCard, WaterCard, DecisionCard, AlertsCard,
} from "@/components/ModuleCards";

export default function Dashboard() {
  const { connected, frame, state, running, start, stop } = useSocket();

  if (!state) {
    return (
      <main className="loading">
        <p>Connecting to backend…</p>
        <p className="muted">Start it with: cd backend && python app.py</p>
      </main>
    );
  }

  const risk = state.decision.overall_risk;

  return (
    <>
      <header className="header">
        <div className="header-left">
          <div className="logo">🏊</div>
          <div className="header-title">
            <h1>AI Smart Swimming Pool Monitoring System</h1>
            <p>R26-IT-143 — Unified Dashboard · 4 AI Modules</p>
          </div>
        </div>
        <div className="header-right">
          <div className={`risk-badge ${risk.toLowerCase()}`}>{risk}</div>
          <div className="connection-status">
            <span className={`status-dot ${connected ? "connected" : "disconnected"}`} />
            <span>{connected ? "Connected" : "Disconnected"}</span>
          </div>
        </div>
      </header>

      <main className="main-grid">
        <section className="left-col">
          <div className="card camera-card">
            <div className="card-header">
              <h2>
                📷 Live CCTV Feed{" "}
                <span className="sub">(shared by Crowd · Drowning · Garbage)</span>
              </h2>
              <div>
                <button className="btn btn-start" onClick={start} disabled={running}>
                  ▶ Start
                </button>
                <button className="btn btn-stop" onClick={stop} disabled={!running}>
                  ■ Stop
                </button>
              </div>
            </div>
            <div className="camera-container">
              {frame ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={frame} alt="camera feed" />
              ) : (
                <div className="camera-overlay">Press Start to begin monitoring</div>
              )}
            </div>
            <div className="camera-meta">
              <span>FPS: <b>{state.fps}</b></span>
              <span>Frame: <b>{state.frame_num}</b></span>
              <span>Session: <b>{state.session_time}</b></span>
            </div>
          </div>
          <DecisionCard decision={state.decision} />
        </section>

        <section className="right-col">
          <CrowdCard crowd={state.crowd} />
          <DrowningCard drowning={state.drowning} />
          <GarbageCard garbage={state.garbage} />
          <WaterCard wq={state.water_quality} />
          <AlertsCard alerts={state.alerts} />
        </section>
      </main>
    </>
  );
}
