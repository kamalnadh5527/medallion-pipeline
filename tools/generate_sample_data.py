from pathlib import Path

import numpy as np
import pandas as pd


LANDING_DIR = Path(__file__).resolve().parent.parent / "data" / "landing"


def generate_sample_data() -> list[Path]:
    rng = np.random.default_rng(42)
    LANDING_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.DataFrame(
        {
            "product_id": ["P100", "P101", "P102", "P103", "P104"],
            "product_name": [
                "  Wireless Mouse",
                "LAPTOP STAND",
                "USB-C Cable ",
                "coffee mug",
                None,
            ],
            "category": ["Electronics", " electronics ", "ELECTRONICS", "Home", "Home"],
            "unit_price": ["24.99", 39.5, "12.00", 8.75, np.nan],
        }
    )

    stores = pd.DataFrame(
        {
            "store_id": ["S001", "S002", "S003", "S004"],
            "store_name": [" Downtown ", "NORTHSIDE", "northside", None],
            "city": ["New York", " chicago", "CHICAGO ", "Austin"],
            "region": ["East", "MIDWEST", "midwest", "South"],
        }
    )

    sales = pd.DataFrame(
        {
            "sale_id": ["T001", "T002", "T003", "T004", "T005", "T006", "T006"],
            "product_id": ["P100", "P101", "P102", "P103", None, "P104", "P104"],
            "store_id": ["S001", "S002", "S003", "S004", "S001", "S002", "S002"],
            "sale_date": [
                "2026-01-05",
                "01/06/2026",
                "2026/01/07",
                "Jan 08 2026",
                "2026-01-09",
                "09-01-2026",
                "09-01-2026",
            ],
            "quantity": [2, "1", 3, "two", np.nan, "4", "4"],
            "discount": [0, "0.10", 0.05, None, "5%", 0, 0],
            "sales_channel": [" Online", "STORE", "online ", "Store", "ONLINE", "store", "store"],
        }
    )
    sales["quantity"] = sales["quantity"].replace("two", np.nan)
    sales["noise_score"] = rng.choice(["1", 2, 3.5, None], size=len(sales))

    written_paths = []
    for name, frame in (
        ("products.csv", products),
        ("stores.csv", stores),
        ("sales.csv", sales),
    ):
        path = LANDING_DIR / name
        frame.to_csv(path, index=False)
        written_paths.append(path)
        print(path)
    return written_paths


if __name__ == "__main__":
    generate_sample_data()
