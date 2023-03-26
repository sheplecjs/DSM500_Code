import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


@dataclass
class FredFxTimeSeries:
    name: str
    series_id: str
    api_key_env: str = "FRED"
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    data_folder: Path = Path.cwd() / "datasets"

    def __post_init__(self) -> None:
        """Gathers and processes data from the FRED API.

        Checks local data for results of an existing API call. If one doesn't exists,
        the call is made, the response is processed, serialized, and saved to the
        data folder.
        """

        file_path = self.data_folder / (self.name + ".csv")

        def call_api(self) -> Optional[pd.DataFrame]:
            key: str = os.getenv(self.api_key_env)
            url: str = "https://api.stlouisfed.org/fred/series/observations"
            payload: dict = {
                "series_id": self.series_id,
                "api_key": key,
                "file_type": "json",
            }

            with requests.Session() as session:
                try:
                    download: requests.models.Response = session.get(
                        url=url, params=payload
                    )
                    return pd.DataFrame(download.json()["observations"])
                except (requests.ConnectionError, requests.ConnectTimeout) as e:
                    print(f"Issue gathering data: {e}.")

        def process_raw(
            data: pd.DataFrame, drop_cols: tuple = ("realtime_start", "realtime_end")
        ) -> pd.DataFrame:
            try:
                data.set_index("date", inplace=True)
            except KeyError:
                pass

            for key in drop_cols:
                try:
                    data.drop(key, inplace=True, axis=1)
                except KeyError:
                    continue

            data.index = pd.to_datetime(data.index)

            # coerce values as nans from fred are odd
            data["OT"] = pd.to_numeric(data.value, errors="coerce")
            try:
                del data["value"]
            except KeyError:
                pass
            data.dropna(inplace=True)

            return data

        def save_data(path: Path) -> None:
            # create data dir if needed
            try:
                os.mkdir("datasets/")
            except FileExistsError:
                pass

            self.data.to_csv(path)

        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path)
            
        else:
            raw = call_api(self)
            self.data = process_raw(raw)
            save_data(file_path)
