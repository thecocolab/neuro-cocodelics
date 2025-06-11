# Neurocognitive Insights into Altered States of Consciousness

Combining Magnetoencephalography (MEG) and Machine Learning (ML) to explore the neural correlates of altered states of consciousness.

## Setup

1. Create a Python virtual environment and activate it  
   ```sh
   python3.9 -m venv .venv_neurococodelics
   source .venv_neurococodelics/bin/activate
   ```
2. Install project dependencies  
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

All ML experiments are driven by YAML config files in the `machine-learning/configs/` folder.  
Copy an existing config and modify as needed (e.g. data paths, features, models).

Example config:  
```
machine-learning/configs/toy_ml_config.yml
```

## Running Analysis

Use the `run_ml.py` script to launch your pipeline with a chosen config:

```sh
python machine-learning/run_ml.py \
  --config machine-learning/configs/toy_ml_config.yml
```

This will:
- Load your data (e.g. `local_data/ketamine.csv`)
- Select features & targets per the config
- Run analyses (feature selection, hyperparam search, etc.)
- Save results to `results/`, including a global summary at `results/<experiment_id>.pkl`

Inspect logs and outputs in the `results/` directory once complete.