Weibull-Knowledge-Informed-ML
==============================

> Exploring the concept of knowledge-informed machine learning with the use of a Weibull-based loss function. Used to predict remaining useful life (RUL) on the [IMS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing) and [PRONOSTIA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto) (also called FEMTO) bearing data sets.

The experiment is detailed in the Journal of Prognostics and Health Management. The paper also gives an overview of knowledge informed machine learning as it applies to prognostics and health management (PHM). Please cite (link below) this work if you find it useful!

You can replicate the work by following the instructions in the [Setup](#setup) section. If you have any questions, leave a comment in the discussion, or email me (18tcvh@queensu.ca).

In this work, we use the definition of knowledge informed machine learning from von Rueden et al. (their excellent paper is [here](https://arxiv.org/abs/1903.12394)). Here's the general taxonomy of our knowledge informed machine learning experiment:

![source_rep_int](./reports/figures/source_rep_int.png)





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
   * Windows: Manually download the the [IMS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing) and [PRONOSTIA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto) (FEMTO) data sets from NASA prognostics data repository. Put in `data/raw` folder.
   * HPC: use `make download`. Will automatically detect HPC environment.
   
4. Extract raw data.
   * Linux/MacOS: use `make extract`. Will automatically extract to appropriate `data/raw` directory.
   * Windows: Manually extract data. See the [Project Organization](#project-organization) section for folder structure.
   * HPC: use `make download`. Will automatically detect HPC environment. Again, modify for your HPC cluster.
   
5. Ensure virtual environment is activated. `conda activate weibull`

6. From root directory of `weibull-knowledge-informed`, run `pip install -e .` -- this will give the python scripts access to the `src` folders.

7. Train!

   * Linux/MacOS: use `make train_ims` or `make train_femto`. (note: set constants in `train_models.py` for changing random search parameters. Will eventually modify for use with argeparse)

     ...

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

## Future List

Things to add, sometime, in the future:

* 
