import { Card, Stat, Pill } from "./Card";

export function CrowdCard({ crowd }) {
  return (
    <Card title="👥 Crowd Detection" status={crowd.model_status}>
      <div className="stat-row">
        <Stat value={crowd.swimmer_count} label="Swimmers" />
        <Stat value={crowd.density_level} label="Density" />
        <Stat value={crowd.bather_load} label="Bather Load (ph)" />
      </div>
      <div className="recommendation">{crowd.maintenance_recommendation}</div>
    </Card>
  );
}

export function DrowningCard({ drowning }) {
  const danger = drowning.status === "DANGER";
  return (
    <Card title="🚨 Drowning Detection" status={drowning.model_status} flash={danger}>
      <div className="stat-row">
        <Stat value={drowning.swimming} label="Swimming" tone="ok" />
        <Stat value={drowning.drowning} label="Drowning" tone="danger" />
        <Stat value={drowning.out_of_water} label="Out of Water" />
      </div>
      <div className="recommendation">
        <Pill text={drowning.status} tone={danger ? "danger" : "safe"} />
        {"  "}Alerts this session: <b>{drowning.total_alerts}</b>
      </div>
    </Card>
  );
}

export function GarbageCard({ garbage }) {
  return (
    <Card title="🗑 Garbage Detection" status={garbage.model_status}>
      <div className="stat-row">
        <Stat value={garbage.objects_detected} label="Objects" />
        <Stat value={garbage.alert_status} label="Status" />
        <Stat value={garbage.total_events} label="Events" />
      </div>
      <div className="recommendation">
        {garbage.object_labels?.length
          ? `Garbage: ${garbage.object_labels.join(", ")}`
          : "No garbage detected"}
        {garbage.non_garbage_labels?.length
          ? ` · Non-garbage: ${garbage.non_garbage_labels.join(", ")}`
          : ""}
      </div>
    </Card>
  );
}

function statusTone(status) {
  return status === "SAFE" ? "safe" : status === "CRITICAL" ? "danger" : "warn";
}

export function WaterCard({ wq, onStartTest }) {
  const f = (v) => (v === null || v === undefined ? "–" : v);
  const test = wq.test || { status: "idle", elapsed: 0, duration: 60, warmup: 20, samples_collected: 0, result: null };
  const testRunning = test.status === "running";
  const result = test.result;

  return (
    <Card title="💧 Water Quality" status={wq.model_status}>
      <div className="stat-row wq">
        <Stat value={f(wq.ph)} label="pH" />
        <Stat value={f(wq.temperature)} label="Temp °C" />
        <Stat value={f(wq.chlorine)} label="Cl ppm" />
        <Stat value={f(wq.turbidity)} label="NTU" />
        <Stat value={f(wq.tds)} label="TDS" />
      </div>
      <div className="recommendation">
        <Pill text={wq.status} tone={statusTone(wq.status)} />
        {"  "}Sensor: <b>{wq.sensor_connected ? "connected" : "offline"}</b>
      </div>

      <div className="wq-test">
        <button className="btn btn-test" onClick={onStartTest} disabled={testRunning}>
          {testRunning ? `Testing… ${Math.round(test.elapsed)}s / ${test.duration}s` : "▶ Take Final Reading (60s)"}
        </button>

        {testRunning && (
          <>
            <div className="test-progress-bar">
              <div
                className="test-progress-fill"
                style={{ width: `${Math.min(100, (test.elapsed / test.duration) * 100)}%` }}
              />
            </div>
            <div className="muted test-progress-note">
              {test.elapsed < test.warmup
                ? `Warming up… averaging starts at ${test.warmup}s`
                : `Collecting samples… ${test.samples_collected} so far`}
            </div>
          </>
        )}

        {test.status === "complete" && result && (
          <div className={`test-result ${result.status.toLowerCase()}`}>
            <div className="test-result-header">
              Final reading (avg of {result.sample_count} samples):{" "}
              <Pill text={result.status} tone={statusTone(result.status)} />
            </div>
            <div className="stat-row wq">
              <Stat value={f(result.ph)} label="pH" />
              <Stat value={f(result.temperature)} label="Temp °C" />
              <Stat value={f(result.chlorine)} label="Cl ppm" />
              <Stat value={f(result.turbidity)} label="NTU" />
              <Stat value={f(result.tds)} label="TDS" />
            </div>
            {result.reasons?.length > 0 && (
              <ul className="notes">
                {result.reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}

export function DecisionCard({ decision }) {
  const tone =
    decision.maintenance_urgency === "LOW"
      ? "safe"
      : decision.maintenance_urgency === "IMMEDIATE"
      ? "danger"
      : "warn";
  return (
    <Card title="🧠 Decision Engine">
      <div className="recommendation">
        Maintenance urgency: <Pill text={decision.maintenance_urgency} tone={tone} />
      </div>
      <ul className="notes">
        {decision.notes.length ? (
          decision.notes.map((n, i) => <li key={i}>{n}</li>)
        ) : (
          <li className="muted">No cross-module concerns</li>
        )}
      </ul>
    </Card>
  );
}

export function AlertsCard({ alerts }) {
  return (
    <Card title="🔔 Recent Alerts">
      <ul className="alert-list">
        {alerts.length ? (
          alerts.map((a, i) => (
            <li key={i} className={`alert-item ${a.severity}`}>
              <b>{a.time}</b> · {a.module}: {a.message}
            </li>
          ))
        ) : (
          <li className="muted">No alerts yet</li>
        )}
      </ul>
    </Card>
  );
}
