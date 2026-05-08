import { NavLink } from 'react-router-dom';

const navigationItems = [
  { label: 'Crowd Monitoring', to: '/crowd-monitoring' },
  { label: 'Water Quality', to: '/water-quality' },
  { label: 'Drowning Detection', to: '/drowning-detection' },
  { label: 'Garbage Detection', to: '/garbage-detection' },
];

export default function Sidebar() {
  return (
    <aside className="hidden w-72 border-r border-white/10 bg-slate-900/95 px-5 py-6 lg:flex lg:flex-col">
      <div className="mb-8">
        <p className="text-xs uppercase tracking-[0.3em] text-pool-300">AI Monitoring</p>
        <h1 className="mt-2 text-2xl font-semibold text-white">Pool Dashboard</h1>
      </div>

      <nav className="space-y-2">
        {navigationItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              [
                'block rounded-xl px-4 py-3 text-sm transition-colors duration-200',
                isActive ? 'bg-pool-500/20 text-pool-200' : 'text-slate-300 hover:bg-white/5 hover:text-white',
              ].join(' ')
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
