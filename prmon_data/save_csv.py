import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/karthik_g/prmon_data")
FILES = {
    "normal_run1": "normal",
    "normal_run2": "normal",
    "normal_run3": "normal",
    "anomaly_highmem": "anomaly",
    "anomaly_highprocs": "anomaly",
    "anomaly_combined": "anomaly",
}
dfs = []
for name, label in FILES.items():
    with open(DATA_DIR / f"{name}.json") as f:
        df = pd.DataFrame(json.load(f))
    df["run"] = name
    df["label"] = label
    dfs.append(df)

pd.concat(dfs).to_csv(DATA_DIR / "all_data.csv", index=False)
print("Saved all_data.csv")
