# 🚀 How to Run — FloodRisk Sri Lanka C1 Satellite Dashboard

> A plain-language, step-by-step guide to get the dashboard running on your computer.

---

## ✅ Before You Start — Check These

| Requirement | How to Check | Minimum |
|---|---|---|
| **Python** | Open terminal → `python --version` | Python **3.10** or higher |
| **pip** | `pip --version` | Any recent version |
| **Free RAM** | Task Manager → Performance | At least **4 GB** free |
| **Project files** | `c1_satellite/` folder must be present | All subfolders intact |

> **Your system:** Python 3.12.5 ✅ — you're good to go.

---

## 📁 Step 1 — Open the Project Folder

Open **PowerShell** (or Command Prompt) and navigate to the project root:

```powershell
cd "C:\Users\hp\Desktop\J26-DS-314"
```

> 💡 **Tip:** You can also right-click the `J26-DS-314` folder in File Explorer → **"Open in Terminal"**

---

## 🐍 Step 2 — Activate the Virtual Environment

The project already has a `.venv` folder. Just activate it:

```powershell
# PowerShell (recommended)
.venv\Scripts\Activate.ps1
```

```cmd
# OR Command Prompt
.venv\Scripts\activate.bat
```

**You'll know it worked** when you see `(.venv)` at the start of your terminal prompt:

```
(.venv) PS C:\Users\hp\Desktop\J26-DS-314>
```

> ⚠️ **If you see a red error about execution policy**, run this first, then try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 📦 Step 3 — Install Dependencies (First Time Only)

> **Skip this step if you've already installed before.**

```powershell
pip install fastapi "uvicorn[standard]"
pip install -r c1_satellite/backend/requirements-lock.txt
```

This will take 1–3 minutes. You'll see packages downloading and installing.

---

## ▶️ Step 4 — Start the Server

Run this **one command** from the `J26-DS-314` folder (with venv active):

```powershell
.venv\Scripts\python c1_satellite/backend/scripts/run_c1_api.py --host 127.0.0.1 --port 8000
```

### ✅ You should see this output:

```
Starting FloodRisk Sri Lanka C1 API on http://127.0.0.1:8000
Swagger interactive documentation: http://127.0.0.1:8000/docs
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> 🕐 The first startup takes **10–20 seconds** while the model loads.  
> Do **not** close this terminal window — it must stay open while you use the dashboard.

---

## 🌐 Step 5 — Open the Dashboard

Open your browser and go to:

```
http://127.0.0.1:8000/dashboard
```

You should see the **FloodRisk Sri Lanka** dashboard with the logo at the top.

---

## 🖥️ Step 6 — Using the Dashboard

### Tab 1 — Prediction (opens by default)

1. **Select a River Station** from the dropdown (e.g. *Ellagawa — Kalu Ganga*)
2. **Select a Forecast Date** from the calendar (validation dates: 10–16 Dec 2025)
3. **Select a Model** (default: *Model A — Latest SAR + SWI* ← best model)
4. Click the **"Predict & Compare"** button
5. Results appear instantly:
   - 4 KPI cards (Current Stage, Predicted, Actual, Error)
   - Flood Risk badge (Normal / Minor / Major / Extreme)
   - Water level gauge bars
   - Time-series chart
   - Satellite evidence panels (SAR + SWI images)
   - Model comparison table (scroll down)

### Tab 2 — Satellite Data
- Shows how Sentinel-1 SAR and ASCAT SWI data was ingested and quality-controlled

### Tab 3 — Map View
- Interactive map of all 40 DMC monitoring stations
- Click any station marker to see its details

### Tab 4 — About
- Research methodology and system specifications

---

## 🔌 Step 7 — Stop the Server

When you're done, go back to the terminal and press:

```
Ctrl + C
```

The server will shut down cleanly.

---

## 🔁 Next Time You Run It

Just repeat **Step 2** and **Step 4** — that's all:

```powershell
# 1. Activate venv
.venv\Scripts\Activate.ps1

# 2. Start server
.venv\Scripts\python c1_satellite/backend/scripts/run_c1_api.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/dashboard` in your browser.

---

## 🛠️ Common Problems & Fixes

### ❌ "Port 8000 is already in use"

The server is already running in another terminal. Either:
- Find that terminal and press `Ctrl+C` to stop it, OR
- Use a different port:

```powershell
.venv\Scripts\python c1_satellite/backend/scripts/run_c1_api.py --port 8001
```
Then go to `http://127.0.0.1:8001/dashboard`

---

### ❌ "No module named 'fastapi'" or "No module named 'uvicorn'"

The venv is not activated. Look for `(.venv)` in your prompt. If it's missing:

```powershell
.venv\Scripts\Activate.ps1
```

---

### ❌ Dashboard page loads but shows no data / spinner spins forever

1. Check the server terminal for any red error messages
2. Try refreshing the page (`Ctrl + Shift + R`)
3. Open browser DevTools (`F12`) → Console tab → look for red errors
4. Test the API directly: open `http://127.0.0.1:8000/api/v1/health` in browser

---

### ❌ "Activate.ps1 cannot be loaded because running scripts is disabled"

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Press `Y` when prompted, then try activating again.

---

### ❌ Map shows blank tiles

The map uses OpenStreetMap — you need an **internet connection** for the map tiles to load.

---

## 📡 API Quick Test (Optional)

While the server is running, you can test the API in your browser or terminal:

| URL | What it shows |
|-----|--------------|
| `http://127.0.0.1:8000/api/v1/health` | Server health + active model info |
| `http://127.0.0.1:8000/api/v1/stations` | All 40 DMC stations |
| `http://127.0.0.1:8000/docs` | Interactive API documentation (Swagger UI) |

---

## 📂 What's in the Project

```
J26-DS-314/
├── .venv/                    ← Python virtual environment
└── c1_satellite/
    ├── README.md             ← Full technical documentation
    ├── backend/
    │   ├── main.py           ← The API server code
    │   ├── improvements/     ← ML prediction engine
    │   ├── saved_models/     ← Trained model files (.joblib)
    │   ├── local_data/       ← Satellite data (SAR patches, SWI rasters)
    │   └── scripts/
    │       └── run_c1_api.py ← ← ← SERVER ENTRY POINT (run this!)
    └── frontend/
        └── index.html        ← The dashboard UI
```

---

*FloodRisk Sri Lanka · J26-DS-314 · Research Prototype*
