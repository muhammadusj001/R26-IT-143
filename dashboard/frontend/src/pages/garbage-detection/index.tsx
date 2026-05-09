export default function GarbageDetectionPage() {
  return (
    <section className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.25em] text-pool-300">Module 4</p>
        <h3 className="mt-2 text-3xl font-semibold text-white">Garbage Detection</h3>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Detection feed</div>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Cleanup alerts</div>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">History log</div>
      </div>
    </section>
  );
}
