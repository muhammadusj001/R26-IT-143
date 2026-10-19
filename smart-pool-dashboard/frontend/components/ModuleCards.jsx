import { Card, Stat, Pill } from "./Card";
import { BACKEND } from "../app/useSocket";

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

// One-line meanings for the Incidents (i) tooltip; titles come from the
// backend's event_legend when present.
const INCIDENT_MEANINGS = {
  GARBAGE_IN_POOL: ["Garbage on the water surface", "Garbage confirmed floating in the pool."],
  SUBMERGED_GARBAGE: ["Garbage under the water", "Garbage seen below the surface - remove it from the floor."],
  SUNK_OBJECT: ["Object sank", "Something vanished inside the pool (often with a splash) - probably on the floor."],
  THROW_NO_FLOATING_OBJECT: ["Throw seen, nothing floating", "Someone made a throwing movement but nothing floated - it may have sunk."],
  UNDERWATER_OBJECT: ["New object on the pool floor", "Something new has stayed still on the pool floor for several seconds."],
};

const num = (v) => (typeof v === "number" && isFinite(v) ? v : 0);

function IncidentsLabel({ legend }) {
  const types = Object.keys(INCIDENT_MEANINGS);
  Object.keys(legend).forEach((t) => { if (!types.includes(t)) types.push(t); });
  return (
    <>
      Incidents
      <span className="info-icon" tabIndex={0} aria-label="What do incidents mean?">
        i
        <span className="info-tip" role="tooltip">
          <b className="info-tip-head">Incident = one confirmed garbage event</b>
          {types.map((t) => {
            const title = legend[t] || (INCIDENT_MEANINGS[t] || [t])[0];
            const meaning = (INCIDENT_MEANINGS[t] || [])[1];
            return (
              <span key={t} className="info-tip-row">
                <b>{title}</b>{meaning ? ` — ${meaning}` : ""}
              </span>
            );
          })}
        </span>
      </span>
    </>
  );
}

function IncidentItem({ ev }) {
  const risk = String(ev.risk || "").toUpperCase();
  const riskTone = risk === "HIGH" ? "danger" : risk === "MEDIUM" ? "warn" : "";
  const pos = ev.position === "underwater" ? "under water" : ev.position || "";
  const meta = [ev.object, pos].filter(Boolean).join(" · ");
  const snap = ev.snapshot
    ? `${BACKEND}/api/garbage/snapshot/${encodeURIComponent(ev.snapshot)}`
    : null;
  return (
    <li className="incident-item" title={ev.explanation || ev.message || ""}>
      {snap && (
        <a className="incident-thumb" href={snap} target="_blank" rel="noopener noreferrer">
          <img
            src={snap}
            alt="incident snapshot"
            loading="lazy"
            onError={(e) => { e.currentTarget.parentNode.style.display = "none"; }}
          />
        </a>
      )}
      <div className="incident-body">
        <div className="incident-top">
          <span className="incident-time">{ev.time || ""}</span>
          <b className="incident-title">{ev.title || ev.type || "Incident"}</b>
          {risk && <Pill text={risk} tone={riskTone} />}
        </div>
        {meta && <div className="incident-meta">{meta}</div>}
        {ev.intent_text && <div className="incident-intent">{ev.intent_text}</div>}
        {ev.reason && <div className="incident-reason">{ev.reason}</div>}
      </div>
    </li>
  );
}

export function GarbageCard({ garbage }) {
  const legend =
    garbage.event_legend && typeof garbage.event_legend === "object" ? garbage.event_legend : {};
  const feed = Array.isArray(garbage.event_feed)
    ? garbage.event_feed.filter((ev) => ev && typeof ev === "object").slice(0, 5)
    : [];
  const action =
    typeof garbage.human_action === "string" && garbage.human_action ? garbage.human_action : "–";
  const throwing = action.toUpperCase().startsWith("THROWING");
  return (
    <Card title="🗑 Garbage Detection" status={garbage.model_status}>
      <div className="stat-row">
        <Stat value={garbage.objects_detected} label="Objects" />
        <Stat value={garbage.alert_status} label="Status" />
        <Stat value={garbage.total_events} label={<IncidentsLabel legend={legend} />} />
      </div>
      <div className="garbage-lines">
        <div>
          On surface: <b>{num(garbage.surface_count)}</b>{"  |  "}
          Under water: <b>{num(garbage.underwater_count)}</b>
        </div>
        <div>
          Human action: <b className={`human-action ${throwing ? "throwing" : ""}`}>{action}</b>
        </div>
      </div>
      <div className="recommendation">
        {garbage.object_labels?.length
          ? `Garbage: ${garbage.object_labels.join(", ")}`
          : "No garbage detected"}
        {garbage.non_garbage_labels?.length
          ? ` · Non-garbage: ${garbage.non_garbage_labels.join(", ")}`
          : ""}
      </div>
      <div className="incident-feed">
        <div className="incident-feed-title">Recent garbage incidents</div>
        <ul className="incident-list">
          {feed.length ? (
            feed.map((ev, i) => <IncidentItem key={`${ev.time}-${ev.type}-${i}`} ev={ev} />)
          ) : (
            <li className="muted">No incidents yet</li>
          )}
        </ul>
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
        {"  "}Sensor: <b>{wq.simulated ? "simulated" : wq.sensor_connected ? "connected" : "offline"}</b>
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
