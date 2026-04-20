import pandas as pd
from pathlib import Path

INPUT = Path(__file__).parent.parent / "data/Nibor-Summary-Statistics-March-2026.xlsx"
OUTPUT = Path(__file__).parent.parent / "data/nibor_monthly.csv"

# Les rådata uten header
df = pd.read_excel(INPUT, sheet_name="Summary statistics", header=None)

# Hent kolonne A (dato) og kolonne C (1-måneds NIBOR), hopp over de 3 første header-radene
nibor = df.iloc[3:, [0, 2]].copy()
nibor.columns = ["date", "nibor_1m"]

# Fjern rader uten dato (NaN)
nibor = nibor.dropna(subset=["date"])

# Formater dato til YYYY-MM
nibor["date"] = pd.to_datetime(nibor["date"]).dt.strftime("%Y-%m")

# Sorter kronologisk
nibor = nibor.sort_values("date").reset_index(drop=True)

nibor.to_csv(OUTPUT, index=False)
print(nibor.head(10).to_string())
print(f"\nSaved {len(nibor)} rows to {OUTPUT}")