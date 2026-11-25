"""Execute the final notebooks in order and write the executed copies (with outputs) in place.

    python run_all.py                 # tasks 1, 2, 3, 5, Task 4 ensemble, then make_submission
    python run_all.py task1 task5     # only some of them

Needs: CPE342_DATA (Kaggle download folder). Task 4 ensemble also needs the saved member probabilities (see task4/README.md).
Each notebook runs with its own folder as working directory and writes its results to outputs/ (next to this file).
"""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ORDER = {"task1": "task1/task1.ipynb", "task2": "task2/task2.ipynb", "task3": "task3/task3.ipynb", "task5": "task5/task5.ipynb",
         "task4": "task4/task4_ensemble.ipynb", "make_submission": "make_submission/make_submission.ipynb"}
todo = sys.argv[1:] or list(ORDER)
for key in todo:
    path = HERE / ORDER[key]
    nb = nbformat.read(path, as_version=4)
    print(f"=== running {path.relative_to(HERE)}", flush=True)
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
print("done")
