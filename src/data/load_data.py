import pandas as pd
from pathlib import Path

def get_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Project root not found. Make sure 'src' folder exists.")    

def load_data(data_dir = None):
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test