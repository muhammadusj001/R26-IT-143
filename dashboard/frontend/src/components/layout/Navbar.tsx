export default function Navbar() {
  return (
    <header className="border-b border-white/10 bg-slate-900/70 px-6 py-4 backdrop-blur-md lg:px-8">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Real-Time Control Center</p>
          <h2 className="mt-1 text-lg font-medium text-white">Smart Swimming Pool Monitoring System</h2>
        </div>
        <div className="rounded-full border border-pool-400/30 bg-pool-400/10 px-4 py-2 text-sm text-pool-200">
          Live Dashboard Ready
        </div>
      </div>
    </header>
  );
}
