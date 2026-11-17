# House price pipeline

A small linear regression that predicts the price of a house (thousand €) from
its size, rooms and age, with a GitHub Actions pipeline that tests, trains,
checks and releases it, and, for every version tag, publishes it on Hugging
Face and updates the app that uses it.

| File | What it does |
|---|---|
| `data/houses.csv` | 400 synthetic houses: the training data |
| `train.py` | trains the model, measures it on 20% of the houses, saves `model.joblib` and `metrics.json` |
| `check_quality.py` | the quality gate: fails if the error is too large or no better than the baseline; also asks the model for the price of three example houses, shown in the run's Summary and in the release notes |
| `tests/test_data.py` | tests on the data, run before training |
| `tests/test_model.py` | tests on the answers of the trained model, run after training |
| `requirements.txt` | the libraries, at fixed versions |
| `.github/workflows/ml-pipeline.yml` | the pipeline GitHub runs on every push |
| `space/` | the app on Hugging Face (`app.py`, `requirements.txt`, and `README.md`, which fixes the Gradio and Python versions of the Space), for a Space with ZeroGPU hardware; the pipeline writes the model's repository and version into it before every upload |

## Run it on your computer

Use Python 3.13, the same version as the pipeline. First create a virtual
environment (venv): a folder, `.venv`, with its own copy of Python and its own
libraries, so that the versions in `requirements.txt` do not mix with anything
else installed on your computer. `.gitignore` keeps `.venv` out of the repository.

Windows (PowerShell or Command Prompt):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -v tests/test_data.py
.venv\Scripts\python train.py
.venv\Scripts\python -m pytest -v tests/test_model.py
.venv\Scripts\python check_quality.py
```

macOS or Linux:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -v tests/test_data.py
.venv/bin/python train.py
.venv/bin/python -m pytest -v tests/test_model.py
.venv/bin/python check_quality.py
```

Each command calls the Python inside `.venv` directly, so there is nothing to
activate. On GitHub, the pipeline does not need a venv: every run starts on a
new, empty computer that is erased afterwards.

The data is synthetic: generated from a known formula with noise, for learning only.

## Hugging Face

For a tag such as `v1.0`, two more steps run after the release. The first
uploads `model.joblib` and `metrics.json` to the model repository
`<your Hugging Face name>/house-price-model` and tags it with the same version.
The second writes that repository and version into `space/` and uploads the
folder to the Space `<your Hugging Face name>/house-price-app`, which restarts
and loads that version of the model. The first upload creates the model
repository; the Space must be created beforehand on huggingface.co, with
Gradio and ZeroGPU hardware. Both steps need the repository secret `HF_TOKEN`: a Hugging Face token
with permission to write.
