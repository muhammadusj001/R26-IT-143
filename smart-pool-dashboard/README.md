# Smart Pool Dashboard (R26-IT-143) — Integration Repo

ONE dashboard integrating FOUR independent component repos.

```
smart-pool-dashboard/
├── backend/            Flask-SocketIO gateway + shared camera + decision engine
├── components/         the 4 component repos (git submodules)
├── frontend/           Next.js dashboard (production frontend)
└── frontend-vanilla/   plain HTML/JS dashboard (zero-build fallback)
```

## Architecture: how 4 repos become 1 dashboard

Each member owns ONE GitHub repo containing an importable Python package
+ models + ml + standalone demo (viva-checkable alone). This repo mounts
them as git submodules under `components/` and the backend imports their
packages directly. The shared camera loop feeds one frame to Crowd,
Drowning and Garbage; Water Quality runs on its own thread; the Decision
Engine is the only place module outputs are combined.

## GitHub setup (do once)

Each member creates their repo and pushes their folder:
```bash
cd component1-crowd-maintenance     # (each member: their own folder)
git init && git add . && git commit -m "Component 1: crowd-aware maintenance"
git branch -M main
git remote add origin https://github.com/<member1>/component1-crowd-maintenance.git
git push -u origin main
```
Repeat for components 2, 3, 4 (each on that member's account).

Then ONE member (or a team org) creates the integration repo:
```bash
cd smart-pool-dashboard
rm -rf components/*                 # remove the local copies
git init && git add . && git commit -m "Integration dashboard"
git submodule add https://github.com/<member1>/component1-crowd-maintenance.git components/component1-crowd-maintenance
git submodule add https://github.com/<member2>/component2-water-quality.git       components/component2-water-quality
git submodule add https://github.com/<member3>/component3-drowning-detection.git  components/component3-drowning-detection
git submodule add https://github.com/<member4>/component4-garbage-detection.git   components/component4-garbage-detection
git commit -m "Add component submodules"
git push -u origin main
```
Anyone cloning later: `git clone --recurse-submodules <dashboard-repo-url>`
Pull component updates: `git submodule update --remote`

(This delivered zip ships plain copies inside components/ so it runs
immediately without any git setup.)

## Run

Backend:
```bash
pip install -r requirements.txt
cd backend && python app.py          # simulation mode by default
```

Frontend (choose one):
```bash
# A) Next.js (production frontend)
cd frontend && npm install && npm run dev    # http://localhost:3000
# B) Zero-build fallback — just open http://localhost:5000
```

Real hardware: see `.env.example` (CAMERA_SOURCE, SENSOR_SOURCE).

## Stack decision (asked: "Python backend + Next.js — good?")
Yes. Flask-SocketIO backend (Component 3's proven stack) + Next.js
consuming it via socket.io-client is professional and panel-impressive.
The vanilla frontend is kept as a guaranteed-working fallback for demo
day — if Node/npm misbehaves on the presentation laptop, port 5000 still
shows the full dashboard.
