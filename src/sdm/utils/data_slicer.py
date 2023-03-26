import csv
import pandas as pd
import glob
from typing import List

def provide_slices(dir: str, slices: List[float] = [0.9, 0.8, 0.7, 0.6, 0.5]):
    datasets = [path for path in glob.iglob(f"{dir}/*.csv")]
    for s in slices:
        for d in datasets:
            flag = str(s).replace(".", "")
            df = pd.read_csv(d)
            records = len(df)
            sliced = df.iloc[: int(records * s)]
            sliced.to_csv(f"{d}_{flag}.csv")
