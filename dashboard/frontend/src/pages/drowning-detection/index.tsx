export default function DrowningDetectionPage() {
  return (
    <section className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.25em] text-pool-300">Module 3</p>
        <h3 className="mt-2 text-3xl font-semibold text-white">Drowning Detection</h3>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Pose stream</div>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Risk signals</div>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">Emergency alerts</div>
      </div>
    </section>
  );
}
