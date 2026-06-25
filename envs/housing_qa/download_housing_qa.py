import json
import zipfile
from io import TextIOWrapper

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_housing_qa(name: str):
    files = {
        "questions": "data/questions.json.zip",
        "questions_aux": "data/questions_aux.json.zip",
        "statutes": "data/statutes.tsv.zip",
    }

    zip_path = hf_hub_download(
        repo_id="reglab/housing_qa",
        repo_type="dataset",
        filename=files[name],
    )

    with zipfile.ZipFile(zip_path) as z:
        if name in {"questions", "questions_aux"}:
            with z.open(f"{name}.json") as f:
                rows = json.load(TextIOWrapper(f, encoding="utf-8"))
            return Dataset.from_list(rows)

        if name == "statutes":
            with z.open("statutes.tsv") as f:
                df = pd.read_csv(f, sep="\t")
            return Dataset.from_pandas(df, preserve_index=False)

    raise ValueError(f"Unknown housing_qa config: {name}")

if __name__ == "__main__":
    for name in ["questions", "questions_aux", "statutes"]:
        dataset = load_housing_qa(name)
        dataset.save_to_disk(f"{DATA_DIR}/housing_qa_{name}")
    
    