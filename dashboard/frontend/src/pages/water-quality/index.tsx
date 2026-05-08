export default function WaterQualityPage() {
  return (
    <section className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.25em] text-pool-300">Module 2</p>
        <h3 className="mt-2 text-3xl font-semibold text-white">Water Quality</h3>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">pH overview</div>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Temperature trend</div>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Chlorine status</div>
      </div>
    </section>
  );
}
