# Contribute

## Development setup

```bash
git clone https://github.com/LLR-ILD/alldecays.git
cd alldecays
git checkout -b my_improvements master
```

Now you have your own branch to try out new stuff.

## venv

```bash
python -m venv py3  # This folder was set up to be ignored by git.
source py3/bin/activate
pip install -r requirements-dev.txt
```

The python environment by removing the `py3` folder.

## conda

```bash
conda create -c conda-forge --name alldecays-dev --file requirements-dev.txt
conda activate alldecays-dev
```

Deactivate and remove in the standard way:

```bash
conda deactivate
conda env remove -n alldecays-dev
```

## Check your work

From the root of the git repository.

* An in-place build:

    ```bash
    python -m pip install -e .
    ```

* Ensure the required code style:

    ```bash
    pre-commit install
    ```

    This will run `black` over your code each time you attempt to make a commit and warn you if there is an error, canceling the commit.

* Run the tests:

    ```bash
    python -m pytest
    ```
