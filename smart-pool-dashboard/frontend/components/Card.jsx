export function Card({ title, status, children, flash }) {
  return (
    <div className={`card ${flash ? "flash" : ""}`}>
      <div className="card-header">
        <h2>{title}</h2>
        {status && <span className="model-status">{status}</span>}
      </div>
      {children}
    </div>
  );
}

export function Stat({ value, label, tone }) {
  return (
    <div className="stat">
      <span className={`stat-val ${tone || ""}`}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

export function Pill({ text, tone }) {
  return <span className={`pill ${tone}`}>{text}</span>;
}
