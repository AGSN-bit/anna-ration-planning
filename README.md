# ANNA — Ration Planning Prototype

ANNA is a local demonstration of how a future ration logistics platform might connect demand forecasts, usable inventory, expiry-aware picking, replenishment planning, and disruption scenarios. This repository contains a small interactive prototype intended for discussion and presentation.

> **Prototype boundary:** All demand, stock, expiry, and scenario values are generated or illustrative. This is not an official Army system and does not represent verified Army processes or operating parameters. It does not use operational data, connect to external systems, place orders, or replace human planning and approval.

## What the prototype demonstrates

- A selectable ration category, with quantities kept in that item's own unit.
- A transparent moving-average benchmark and an optional Random Forest demand forecast.
- A chronological 28-day holdout view with MAE, RMSE, and WAPE calculated from the synthetic series.
- Synthetic batch inventory, days-to-expiry visibility, and a First Expired, First Out (FEFO) pick preview.
- A simplified safety-stock and replenishment calculation that responds to the selected scenario.
- Side-by-side baseline and what-if projections for changed demand, additional delivery delay, and interruption pressure.
- A roadmap view that separates today's demo from later validation and the longer-term multi-depot vision.
- CSV downloads for the forecast and scenario projection.

## Requirements

- Python and `pip`.
- A modern browser (the app runs locally and opens in your browser).
- Internet access for the first dependency installation, unless the required Python packages are already cached.

The dependencies are listed in [`requirements.txt`](requirements.txt). A virtual environment is recommended so the app's packages stay separate from system Python.

### Get the project

Install Git if needed, then clone the repository and enter its folder. If the repository is private, sign in to GitHub and make sure your account has access before cloning.

```bash
git clone https://github.com/AGSN-bit/anna-ration-planning.git
cd anna-ration-planning
```

## Run locally

The app listens on your own computer at `http://localhost:8501` by default. Run these commands from the repository folder. Keep the terminal open while you use the app; stop it with **Ctrl+C**.

### CachyOS and other Linux distributions

Open a terminal and clone the project. Then create and install into a virtual environment. These commands work in Bash and Zsh:

```bash
git clone https://github.com/AGSN-bit/anna-ration-planning.git
cd anna-ration-planning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If your terminal uses **fish** (as on the CachyOS setup shown during development), activate it with the fish-specific script:

```fish
git clone https://github.com/AGSN-bit/anna-ration-planning.git
cd anna-ration-planning
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

To skip activation in fish or another shell, invoke the virtual environment's programs directly:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

### macOS

Open Terminal, clone the repository, enter its folder, then run:

```bash
git clone https://github.com/AGSN-bit/anna-ration-planning.git
cd anna-ration-planning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

Open PowerShell and clone the repository. Create the environment, activate it, install dependencies, and start Streamlit:

```powershell
git clone https://github.com/AGSN-bit/anna-ration-planning.git
cd anna-ration-planning
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation because of its execution policy, you can avoid activation and run the virtual environment executables directly:

```powershell
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\streamlit.exe run app.py
```

You can also open the cloned folder in File Explorer, right-click inside it, and choose **Open in Terminal**.

### Windows Command Prompt

```bat
git clone https://github.com/AGSN-bit/anna-ration-planning.git
cd anna-ration-planning
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Open the local app

After the `streamlit run app.py` command starts, Streamlit prints a **Local URL**, normally:

```text
http://localhost:8501
```

Open that address in Firefox, Chrome, Edge, or another modern browser on the same computer. If port 8501 is already in use, Streamlit will select and print another local port; use the address shown in your terminal.

On the next run, open a terminal in the project directory, activate the existing environment for your shell, and start Streamlit again:

```bash
# Linux/macOS with Bash or Zsh
source .venv/bin/activate
streamlit run app.py
```

```fish
# CachyOS/Linux with fish
source .venv/bin/activate.fish
streamlit run app.py
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Common problems

- **`streamlit: command not found` / not recognized:** activate `.venv` in the current terminal, or run it directly as `.venv/bin/streamlit` on Linux/macOS, `\.venv\Scripts\streamlit.exe` in PowerShell, or `.venv\Scripts\streamlit.exe` in Command Prompt.
- **Activation script parse error in fish:** use `source .venv/bin/activate.fish`, not `source .venv/bin/activate`.
- **`python` is not found on macOS:** use `python3` to create the environment (`python3 -m venv .venv`). On Windows, try the Python launcher `py`.
- **Package installation fails:** check that Python and `pip` are installed and that the computer can reach the package index. Re-run the install command from the project directory with the virtual environment active.
- **Browser page does not load:** check that the Streamlit process is still running and open the exact Local URL printed in the terminal.

## Using the demo

1. Use the sidebar to choose a synthetic-data seed, ration category, forecast method, and planning horizon.
2. Open **Overview** to see usable stock, projected demand, near-expiry quantity, the illustrative replenishment suggestion, and the selected scenario's stock path.
3. Open **Forecast** to inspect synthetic history, future demand, and the generated-series holdout metrics. Download the forecast CSV if useful for a presentation.
4. Open **Inventory & FEFO** to inspect the synthetic batches and see how a seven-day pick is allocated to earlier-expiring eligible batches first.
5. Adjust the what-if controls in the sidebar and compare baseline and disrupted cases in **Overview** or **What-if simulator**.
6. Use **Bigger vision** to explain the intended progression from a focused prototype to validated research and, only if authorized and supported, a broader multi-depot platform.

Changing the seed creates a different reproducible synthetic example. The sliders are teaching controls, not operational parameters. An interruption is represented as temporary demand pressure in this prototype.

## Project structure

```text
ANNA/
├── app.py                 # Streamlit interface and charts
├── requirements.txt       # Python dependencies
├── README.md              # Setup, usage, and scope
└── anna/
    ├── __init__.py
    ├── data.py            # Synthetic demand and batch data
    ├── forecasting.py     # Forecast methods and holdout metrics
    └── inventory.py       # FEFO, replenishment, and scenario replay
```

## Prototype architecture

```text
Generated daily demand ──> Forecast method ──> Demand outlook ─┐
                                                               ├─> Replenishment suggestion
Generated item batches ──> Usable stock + expiry / FEFO ───────┘
                                                                      │
Sidebar scenario controls ──> Baseline / what-if replay <─────────────┘
                                                                      │
                                                         Dashboard + CSV exports
```

The current app is intentionally a local, single-process demonstration. It has no production database, user accounts, role permissions, audit trail, integration layer, multi-depot synchronization, offline workflow, or procurement execution.

## Research and scale-up direction

The prototype is a way to communicate and discuss the workflow, not evidence that a real deployment will improve logistics outcomes. A responsible next step would be to define evaluation questions and compare simple policies across many controlled synthetic scenarios. Any later use of real data or institution-specific workflows would require the relevant authorization, data governance, security review, operational validation, and human approval. A multi-depot platform is a longer-term vision, not a feature delivered by this repository.

## Known limitations

- Generated data does not describe real demand, inventory, storage conditions, transport, or Army procedures.
- Shelf-life and replenishment parameters are illustrative assumptions and must not be used for actual ration handling or procurement decisions.
- Forecast scores describe one synthetic chronological holdout and do not establish real-world accuracy.
- The replenishment rule and disruption simulator are simplified demonstrations, not validated optimization or logistics models.
- Recommendations are for review only; no order is created or transmitted.
