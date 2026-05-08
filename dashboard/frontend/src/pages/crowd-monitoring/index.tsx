import { useSwimmerAnalytics } from '../../hooks/useSwimmerAnalytics';

const densityTheme: Record<'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL', { ring: string; dot: string; label: string }> = {
  LOW: {
    ring: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
    dot: 'bg-emerald-400',
    label: 'Green',
  },
  MEDIUM: {
    ring: 'border-yellow-500/30 bg-yellow-500/10 text-yellow-300',
    dot: 'bg-yellow-400',
    label: 'Yellow',
  },
  HIGH: {
    ring: 'border-orange-500/30 bg-orange-500/10 text-orange-300',
    dot: 'bg-orange-400',
    label: 'Orange',
  },
  CRITICAL: {
    ring: 'border-red-500/30 bg-red-500/10 text-red-300',
    dot: 'bg-red-400',
    label: 'Red',
  },
};

export default function CrowdMonitoringPage() {
  const { data: analyticsData, connected, loading, error } = useSwimmerAnalytics('crowd-dashboard');

  const swimmerCount = analyticsData?.swimmer_count ?? 0;
  const crowdDensity = analyticsData?.crowd_level ?? 'LOW';
  const occupancy = analyticsData?.status?.occupancy_percentage ?? 0;
  const maintenanceUrgency = analyticsData?.maintenance_urgency ?? 'LOW';
  const fps = analyticsData?.fps ?? 0;
  const capacity = analyticsData?.status?.occupancy_capacity ?? 50;
  const streamUrl = import.meta.env.VITE_CAMERA_STREAM_URL ?? 'http://localhost:8000/api/v1/cameras/webcam/stream';
  const maintenance_recommendations = analyticsData?.status?.maintenance_recommendations ?? [];

  const density = densityTheme[crowdDensity];
  const systemHealth = connected ? 'Online and monitoring' : 'Offline - Reconnecting...';
  const healthColor = connected ? 'text-emerald-300' : 'text-yellow-300';

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-pool-300">Module 1</p>
          <h3 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">AI Smart Pool Dashboard</h3>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400 sm:text-base">
            Real-time swimmer occupancy intelligence, live webcam monitoring, and maintenance awareness in one place.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-sm">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">System Health</p>
          <p className={`mt-1 text-sm font-medium ${healthColor}`}>{systemHealth}</p>
          {error && <p className="mt-1 text-xs text-red-400">Error: {error}</p>}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-12">
        <div className="space-y-4 xl:col-span-4">
          <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-slate-950/40 backdrop-blur-md">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm text-slate-400">Live Swimmer Count</p>
                <h4 className="mt-1 text-4xl font-semibold text-white">{loading ? '—' : swimmerCount}</h4>
              </div>
              <div className="rounded-2xl border border-pool-400/20 bg-pool-400/10 px-4 py-2 text-right">
                <p className="text-xs uppercase tracking-[0.25em] text-pool-200">Capacity</p>
                <p className="text-lg font-semibold text-white">{capacity}</p>
              </div>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Occupancy</p>
                <p className="mt-2 text-2xl font-semibold text-white">{loading ? '—' : Math.round(occupancy)}%</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">FPS</p>
                <p className="mt-2 text-2xl font-semibold text-white">{loading ? '—' : Math.round(fps)}</p>
              </div>
            </div>
          </div>

          <div className={`rounded-3xl border p-5 shadow-2xl shadow-slate-950/40 ${density.ring}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.25em] opacity-80">Crowd Density Indicator</p>
                <h4 className="mt-2 text-2xl font-semibold text-white">{loading ? '—' : crowdDensity}</h4>
              </div>
              <span className={`h-4 w-4 rounded-full ${density.dot}`} />
            </div>

            <div className="mt-5 space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-300">
                <span>Status Color</span>
                <span>{density.label}</span>
              </div>
              <div className="h-3 rounded-full bg-white/10">
                <div className="h-3 rounded-full bg-gradient-to-r from-pool-400 via-sky-400 to-cyan-300" style={{ width: `${loading ? 0 : occupancy}%` }} />
              </div>
              <p className="text-sm text-slate-300">
                Green = LOW, Yellow = MEDIUM, Orange = HIGH, Red = CRITICAL
              </p>
            </div>
          </div>
        </div>

        <div className="xl:col-span-5">
          <div className="overflow-hidden rounded-3xl border border-white/10 bg-slate-900/80 shadow-2xl shadow-slate-950/40">
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Real-Time Webcam Feed</p>
                <h4 className="mt-1 text-lg font-semibold text-white">Live Pool Camera</h4>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-sm text-emerald-300">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                Streaming
              </div>
            </div>

            <div className="relative aspect-video bg-black">
              <img
                src={streamUrl}
                alt="Live webcam feed"
                className="h-full w-full object-cover"
                onError={(event) => {
                  const target = event.currentTarget;
                  target.src = 'data:image/svg+xml;charset=UTF-8,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720"%3E%3Crect width="1280" height="720" fill="%230f172a"/%3E%3Ctext x="50%25" y="46%25" fill="%2394a3b8" font-size="34" text-anchor="middle" font-family="Arial, sans-serif"%3EWebcam feed unavailable%3C/text%3E%3Ctext x="50%25" y="54%25" fill="%2364748b" font-size="18" text-anchor="middle" font-family="Arial, sans-serif"%3ECheck the backend stream endpoint%3C/text%3E%3C/svg%3E';
                }}
              />

              <div className="absolute left-4 top-4 rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 text-xs tracking-[0.25em] text-slate-200 backdrop-blur-sm">
                LIVE FEED
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4 xl:col-span-3">
          <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-slate-950/40">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Maintenance Alert</p>
            <div className="mt-3 flex items-start justify-between gap-4">
              <div>
                <h4 className="text-xl font-semibold text-white">Attention Required</h4>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  The current crowd load suggests maintenance should be monitored closely.
                </p>
              </div>
              <div className="rounded-2xl border border-yellow-500/20 bg-yellow-500/10 px-3 py-2 text-sm font-medium text-yellow-300">
                {loading ? '—' : maintenanceUrgency}
              </div>
            </div>

            <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-sm text-slate-300">Suggested next action</p>
              <p className="mt-2 text-base font-medium text-white">
                {loading ? 'Loading...' : maintenance_recommendations.length > 0 ? `${maintenance_recommendations[0].action} (${maintenance_recommendations[0].priority})` : 'No immediate actions needed.'}
              </p>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-slate-950/40">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Live Summary</p>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                <span>Density Status</span>
                <span className={density.dot.replace('bg-', 'text-')}> {loading ? '—' : crowdDensity}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                <span>Webcam FPS</span>
                <span className="text-white">{loading ? '—' : Math.round(fps)}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                <span>Occupancy</span>
                <span className="text-white">{loading ? '—' : Math.round(occupancy)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {maintenance_recommendations.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-3">
          {maintenance_recommendations.slice(0, 3).map((rec) => {
            const toneMap: Record<string, string> = {
              CRITICAL: 'text-red-300',
              HIGH: 'text-orange-300',
              MEDIUM: 'text-yellow-300',
              LOW: 'text-emerald-300',
            };
            return (
              <div key={rec.action} className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-lg shadow-slate-950/20 backdrop-blur-sm">
                <p className={`text-sm font-medium ${toneMap[rec.priority] ?? 'text-slate-300'}`}>{rec.action}</p>
                <p className="mt-2 text-sm leading-6 text-slate-400">Priority: {rec.priority} • Overdue: {rec.overdue_by.toFixed(2)} person-hours</p>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
