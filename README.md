
# Smart Engine – Turbofan RUL (C-MAPSS) 🛠️

Production-ready MLOps project for **Remaining Useful Life (RUL)** prediction on turbofan engines using the NASA **C-MAPSS** dataset.
Built with **ZenML** for pipeline orchestration and **MLflow** for experiment tracking.

> **Status:** ✅ Training pipeline works • 🚧 Deployment not enabled yet

---

## ✨ Highlights

* **Clean, modular steps** (ingest → label RUL → clean → feature engineer → split → train → evaluate)
* **Swappable models** (e.g., Histogram Gradient Boosting, Linear Regression)
* **Experiment tracking** (MLflow autologging ready)
* **Extensible** design with strategy/factory patterns and ZenML pipelines

---

## 🧭 Project Structure

```
Smart_Engine/
├─ data/                     # (ignored) place your raw C-MAPSS archive here
├─ pipelines/
│  ├─ training_pipeline.py   # defines the training pipeline
│  └─ deployment_pipeline.py # (present) deployment WIP / not used now
├─ steps/
│  ├─ data_ingestion_step.py
│  ├─ rul_labeling_step.py
│  ├─ handle_missing_values_step.py
│  ├─ feature_engineering_step.py
│  ├─ outlier_detection_step.py
│  ├─ data_splitter_step.py
│  ├─ model_building_step.py
│  └─ model_evaluator_step.py
├─ src/
│  ├─ ingest_data.py         # robust ZIP/dir ingestion utilities
│  ├─ data_splitter.py
│  ├─ feature_engineering.py
│  └─ model_building.py
├─ run_pipeline.py           # CLI entrypoint to run the training pipeline
├─ requirements.txt
├─ README.md                 # this file
└─ .gitignore
```

---

## 🧰 Tech Stack

* **Python** (3.10 recommended)
* **ZenML** – pipeline orchestration, artifact lineage
* **MLflow** – autologging & experiment tracking
* **scikit-learn** – feature transforms & models
* **pandas / numpy** – data handling

---

## 🚀 Quickstart

> Works on Windows (PowerShell) or WSL/Linux – commands shown for both where relevant.

1. **Create & activate a virtual environment**

**Windows (PowerShell):**

```powershell
cd C:\Users\msi\Downloads\Smart_Engine
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**WSL/Linux:**

```bash
cd ~/Smart_Engine
python3 -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **(Optional) Initialize ZenML (first time only)**

```bash
zenml init
zenml stack describe
```

> The default local stack is enough to run training. You don’t need to add an experiment tracker to get started. If MLflow is available, autologging kicks in.

4. **Place dataset**

* Put your raw **C-MAPSS** archive under `data/raw/` (e.g., `data/raw/archive.zip`).

5. **Run the training pipeline**

```bash
python run_pipeline.py
```

You should see ZenML pipeline logs for each step and a final evaluation summary.

---

## 🧪 Viewing Experiments (Optional)

If MLflow tracking is active, you can open the UI:

```bash
mlflow ui
```

Then navigate to the printed URL (default: [http://127.0.0.1:5000](http://127.0.0.1:5000)) to browse runs, params, and metrics.

---

## 🔗 Pipeline – End to End

**Training pipeline** (implemented & working):

1. **Data Ingestion** – reads `.zip` or directory; merges columns into a tidy DataFrame
   *File:* `steps/data_ingestion_step.py` → uses `src/ingest_data.py`

2. **RUL Labeling** – computes target `RUL = max(cycle per engine) - cycle`
   *File:* `steps/rul_labeling_step.py`

3. **Handle Missing Values** – strategies like mean/median (configurable)
   *File:* `steps/handle_missing_values_step.py`

4. **Feature Engineering** – scaling / transforms (standard, minmax, log, one-hot)
   *File:* `steps/feature_engineering_step.py` & `src/feature_engineering.py`

5. **Outlier Detection** – optional outlier removal (commonly on `RUL`)
   *File:* `steps/outlier_detection_step.py`

6. **Data Split** – train/test split with a configurable target column (`RUL`)
   *File:* `steps/data_splitter_step.py` & `src/data_splitter.py`

7. **Model Building** – choose model via param (e.g., `hgb` or `linreg`)
   *File:* `steps/model_building_step.py` & `src/model_building.py`
   *Notes:* MLflow autologging enabled

8. **Model Evaluation** – computes regression metrics (e.g., MSE)
   *File:* `steps/model_evaluator_step.py`

---

## ⚙️ Configuration & Parameters

Most steps are parameterized via the pipeline entrypoint (e.g., in `run_pipeline.py`):

* `file_path` – path to C-MAPSS archive / folder
* `subset` – C-MAPSS subset (`FD001..FD004`)
* `missing_value_strategy` – e.g., `mean`
* `fe_strategy` – `standard_scaling`, `minmax_scaling`, `log`, `onehot_encoding`
* `algorithm` – `hgb` (recommended) or `linreg`

You can hardcode defaults or wire up CLI args for convenience.

---

## 🧱 Design Notes

* **ZenML Pipelines & Steps**
  Each transformation is a **@step**; the overall workflow is a **@pipeline**. This enforces clean boundaries, cached artifacts, and reproducibility.

* **Strategy Pattern**
  The model builder and feature engineering use strategies to easily swap algorithms / transforms without touching the pipeline code.

* **Experiment Tracking**
  MLflow autologging inside `model_building_step` records params, metrics, and models if an experiment tracker is configured.

* **Artifact Lineage**
  ZenML tracks inputs/outputs across steps for lineage and reproducibility.

---

## 🧭 Roadmap

* ✅ Training pipeline stable
* 🚧 **Deployment pipeline** (MLflow model serving) – **not enabled yet**
* ⏭️ Add model registry, CI/CD, drift monitoring, batch/online inference

---

## 🐞 Troubleshooting

* **“No Python at '/usr/bin\python.exe'” on Windows**
  Your terminal is pointing to a non-existent interpreter. Re-select your venv in VS Code (`Python: Select Interpreter`) and re-run.

* **Pip resolver / version conflicts**
  Pin Python to **3.10**, then reinstall in a fresh venv.

* **MLflow UI not showing runs**
  Ensure MLflow is installed and the run actually executed the training step with autologging (check terminal logs).

---

## 📜 License & Dataset

* Code: **Apache-2.0** (see `LICENSE` if present)
* Data: **NASA C-MAPSS** (respect original dataset terms)

---

## 🤝 Contributing

Issues and PRs are welcome. Please keep steps modular and add tests where possible.

---

## 📦 Reproduce in 60 seconds

```bash
git clone https://github.com/Bertal-Youssef/Smart_Engine.git
cd Smart_Engine
python -m venv .venv && . .venv/Scripts/activate  # Windows
# or: python3 -m venv .venv && source .venv/bin/activate  # Linux/WSL

pip install -r requirements.txt
zenml init
python run_pipeline.py
```

> Deployment is coming next. For now, enjoy clean, reproducible training runs.
