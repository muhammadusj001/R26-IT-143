# R26-IT-143: AI-Based Smart Swimming Pool Monitoring System

This is the repository for the SLIIT Final Year Project.
# Terminal 1 — backend
cd smart-pool-dashboard/backend
source ../venv/bin/activate
PORT=5051 python app.py          # or just `python app.py` if port 5000 is free for you

# Terminal 2 — Next.js frontend
cd smart-pool-dashboard/frontend
NEXT_PUBLIC_BACKEND_URL=http://localhost:5051 npm run dev
