"""Generate the two demo datasets in sample_data/.

The data is synthetic and contains deliberate problems (duplicates, numbers
stored as text, missing values, inconsistent spelling) so the Data Quality tab
has something to find. Run:  python scripts/make_demo_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
rng = np.random.default_rng(42)


def ro_money(value: float, suffix: str) -> str:
    """Format like a Romanian spreadsheet export: 1.099,00 lei."""
    whole, decimals = f"{value:,.2f}".split(".")
    return f"{whole.replace(',', '.')},{decimals} {suffix}"


def sales(n: int = 2000) -> pd.DataFrame:
    products = {
        "Wireless Mouse": ("Accessories", 89), "Laptop Stand": ("Accessories", 149),
        "USB-C Hub": ("Accessories", 119), "Mechanical Keyboard": ("Accessories", 329),
        'Monitor 27"': ("Displays", 1099), 'Monitor 24"': ("Displays", 749),
        "Webcam HD": ("Video", 239), "Noise-cancelling Headphones": ("Audio", 899),
        "Bluetooth Speaker": ("Audio", 279), "External SSD 1TB": ("Storage", 459),
    }
    cities = ["Bucharest", "Cluj-Napoca", "Timisoara", "Iasi", "Brasov", "Constanta"]
    weights = [0.38, 0.18, 0.13, 0.11, 0.11, 0.09]
    names = list(products)

    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    product = rng.choice(names, n)
    qty = rng.integers(1, 6, n)
    price = np.array([products[p][1] for p in product], dtype=float)
    df = pd.DataFrame({
        "Order ID": np.arange(10001, 10001 + n),
        "Date": pd.Series(dates).dt.strftime("%Y-%m-%d"),
        "City": rng.choice(cities, n, p=weights),
        "Product": product,
        "Category": [products[p][0] for p in product],
        "Qty": qty,
        "Unit price": [ro_money(p, "lei") for p in price],
        "Revenue": qty * price,
    }).sort_values("Date").reset_index(drop=True)

    # add some dirty data
    df.loc[rng.choice(n, 60, replace=False), "Category"] = None
    variants = {"Bucharest": ["bucharest", "BUCHAREST", "Bucharest "], "Iasi": ["IASI", "iasi "]}
    for city, spellings in variants.items():
        idx = df.index[df["City"] == city]
        pick = rng.choice(idx, min(25, len(idx)), replace=False)
        df.loc[pick, "City"] = rng.choice(spellings, len(pick))
    duplicates = df.sample(55, random_state=1)
    return pd.concat([df, duplicates]).sort_values("Order ID").reset_index(drop=True)


def hr(n: int = 600) -> pd.DataFrame:
    departments = {"IT": 9500, "Sales": 6500, "Marketing": 7000, "Finance": 8000, "HR": 6000, "Operations": 5500}
    levels = {"Junior": 0.75, "Mid": 1.0, "Senior": 1.4, "Lead": 1.8}
    dept = rng.choice(list(departments), n, p=[0.3, 0.2, 0.12, 0.13, 0.08, 0.17])
    level = rng.choice(list(levels), n, p=[0.35, 0.35, 0.22, 0.08])
    salary = np.array([departments[d] * levels[l] for d, l in zip(dept, level)]) * rng.normal(1, 0.08, n)
    hire = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3900, n), unit="D")
    df = pd.DataFrame({
        "Employee ID": [f"E{1000 + i}" for i in range(n)],
        "Department": dept,
        "Level": level,
        "City": rng.choice(["Bucharest", "Cluj-Napoca", "Iasi", "Remote"], n, p=[0.55, 0.2, 0.1, 0.15]),
        "Hire date": pd.Series(hire).dt.strftime("%d.%m.%Y"),
        "Monthly salary": [ro_money(round(s, -1), "RON") for s in salary],
        "Performance score": np.round(rng.normal(3.6, 0.6, n).clip(1, 5), 1),
        "Remote days / week": rng.integers(0, 6, n),
    })

    df.loc[rng.choice(n, 45, replace=False), "Performance score"] = np.nan
    idx = df.index[df["Department"] == "IT"]
    df.loc[rng.choice(idx, 20, replace=False), "Department"] = rng.choice(["it", "IT "], 20)
    idx = df.index[df["Department"] == "Sales"]
    df.loc[rng.choice(idx, 12, replace=False), "Department"] = "sales"
    duplicates = df.sample(18, random_state=2)
    return pd.concat([df, duplicates]).sort_values("Employee ID").reset_index(drop=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sales().to_csv(OUT / "sales_demo.csv", index=False)
    hr().to_csv(OUT / "hr_demo.csv", index=False)
    print("Wrote", OUT / "sales_demo.csv", "and", OUT / "hr_demo.csv")
