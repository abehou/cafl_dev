import json
import zipfile
from io import TextIOWrapper
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import trange

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


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

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        if name in {"questions", "questions_aux"}:
            with z.open(f"{name}.json") as f:
                rows = json.load(TextIOWrapper(f, encoding="utf-8"))

            output_path = DATA_DIR / f"{name}.jsonl"
            with output_path.open("w", encoding="utf-8") as out:
                for row in rows:
                    # Add state context to the question; this is not in the original dataset.
                    refined_question = f"(For the state of {row.get('state', 'Unknown')}), {row.get('question', '')}"
                    json_data = dict(row)
                    json_data["question"] = refined_question
                    out.write(json.dumps(json_data, ensure_ascii=False) + "\n")

            return output_path
        
        if name == "statutes":
            print(f"Loading statutes from {zip_path}")
            with z.open("statutes.tsv") as f:
                df = pd.read_csv(f, sep="\t")
            output_path = DATA_DIR / "corpus" / "statutes.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as out:
                for row in trange(len(df), desc="Writing statutes.jsonl"):
                    out.write(json.dumps(df.iloc[row].to_dict(), ensure_ascii=False) + "\n")

            return output_path

    raise ValueError(f"Unknown housing_qa config: {name}")

if __name__ == "__main__":
    for name in ["questions", "questions_aux", "statutes"]:
        path = load_housing_qa(name)
        print(f"Saved {name} to {path}")
    
