Weibull-Knowledge-Informed-ML
==============================

> Exploring the concept of knowledge-informed machine learning with the use of a Weibull-based loss function. Used to predict remaining useful life (RUL) on the [IMS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing) and [PRONOSTIA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto) (also called FEMTO) bearing data sets.



[Setup](#setup)

## Setup

Tested in linux (MacOS should also work). If you run windows you'll have to do much of the environment setup and data download/preprocessing manually.

To reproduce results:

1. Clone this repo - `clone https://github.com/tvhahn/weibull-knowledge-informed.git`
2. Create virtual environment. Assumes that Conda is installed.
   * Linux/MacOS: use command from Makefile - `make create_environment`
   * Windows: from root directory - `conda env create -f envweibull.yml`
   * HPC: `make create_environment` will detect HPC environment and automatically create environment from `make_hpc_venv.sh`. Tested on Compute Canada. Modify `make_hpc_venv.sh` for your own HPC cluster.
3. Download raw data.
   * Linux/MacOS: use `make download`. Will automatically download to appropriate `data/raw` directory.
   * Windows: Manually download the the [IMS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing) and [PRONOSTIA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto) data sets from NASA prognostics data repository. Put in `data/raw` folder.
   * HPC: use `make download`. Will automatically detect HPC environment.
4. Extract raw data.
   * Linux/MacOS: use `make extract`. Will automatically extract to appropriate `data/raw` directory.
   * Windows: Manually extract data. See 
   * HPC: use `make download`. Will automatically detect HPC environment.







**Setup**

From the home directory, run `pip install -e .` so that scripts have access to the `src` files.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands to reproduce work, lik `make data` or `make train_ims`
    ├── README.md          <- The top-level README.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump. Downloaded from the NASA Prognostic repository.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (in progress)
    │
    ├── models             <- Trained models, model predictions, and model summaries
    │   ├── interim        <- Intermediate models that have not analyzed. Output from the random search.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
