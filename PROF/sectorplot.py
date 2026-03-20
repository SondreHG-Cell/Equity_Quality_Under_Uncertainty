from pathlib import Path
from collections import Counter
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


SECTOR_GROUP_MAP = {
    # Energy
    "Oil & Gas": "Energy",
    "Oil & Gas Related Equipment and Services": "Energy",
    "Renewable Energy": "Energy",

    # Financials
    "Banking Services": "Financials",
    "Insurance": "Financials",
    "Investment Banking & Investment Services": "Financials",
    "Investment Holding Companies": "Financials",

    # Industrials
    "Freight & Logistics Services": "Industrials",
    "Machinery, Tools, Heavy Vehicles, Trains & Ships": "Industrials",
    "Construction & Engineering": "Industrials",
    "Aerospace & Defense": "Industrials",
    "Passenger Transportation Services": "Industrials",
    "Professional & Commercial Services": "Industrials",

    # Information Technology
    "Software & IT Services": "Information Technology",
    "Semiconductors & Semiconductor Equipment": "Information Technology",
    "Electronic Equipment & Parts": "Information Technology",
    "Communications & Networking": "Information Technology",

    # Communication Services
    "Telecommunications Services": "Communication Services",
    "Media & Publishing": "Communication Services",

    # Consumer Discretionary
    "Specialty Retailers": "Consumer Discretionary",
    "Diversified Retail": "Consumer Discretionary",
    "Automobiles & Auto Parts": "Consumer Discretionary",

    # Consumer Staples
    "Food & Tobacco": "Consumer Staples",

    # Health Care
    "Pharmaceuticals": "Health Care",
    "Pharmaceutical": "Health Care",
    "Biotechnology & Medical Research": "Health Care",
    "Healthcare Equipment & Supplies": "Health Care",
    "Healthcare": "Health Care",

    # Materials
    "Chemicals": "Materials",
    "Metals & Mining": "Materials",
    "Containers & Packaging": "Materials",

    # Utilities
    "Electric Utilities & IPPs": "Utilities",

    # Real Estate
    "Real Estate Operations": "Real Estate",
}


def hent_sektor_fra_fil(filsti: Path):
    """
    Leser første ark i en xlsx-fil og henter sektor fra celle B5.
    Returnerer teksten i B5, eller None hvis noe feiler.
    """
    try:
        wb = openpyxl.load_workbook(filsti, read_only=True, data_only=True)
        ark = wb.worksheets[0]
        sektor = ark["B5"].value
        wb.close()

        if sektor is None:
            return None

        sektor = str(sektor).strip()
        return sektor if sektor else None

    except Exception as e:
        print(f"Feil ved lesing av {filsti.name}: {e}")
        return None


def map_sektor(sektor):
    """
    Mapper rå sektor til gruppert sektor.
    Hvis sektor ikke finnes i mapping returneres original sektor.
    """
    if sektor is None:
        return None

    sektor = sektor.strip()

    if sektor in SECTOR_GROUP_MAP:
        return SECTOR_GROUP_MAP[sektor]

    lower = sektor.lower()

    if "pharma" in lower:
        return "Health Care"
    if "biotech" in lower:
        return "Health Care"
    if "healthcare" in lower or "health care" in lower:
        return "Health Care"

    return sektor


def lag_barplot(sektor_teller, out_plot: Path):
    """
    Lager et pent liggende barplot for sektorfordeling.
    """
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    sorterte = sorted(sektor_teller.items(), key=lambda x: x[1])
    sektorer = [x[0] for x in sorterte]
    antall = [x[1] for x in sorterte]
    total_firms = sum(antall)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(sektorer) + 2)))

    bars = ax.barh(
        sektorer,
        antall,
        color="#5f6d85",   # dus / tåkete navy
        edgecolor="#3e4758",
        linewidth=0.8
    )

    ax.set_xlabel("Number of firms")
    ax.set_ylabel("")
    ax.set_title(f"Firms by Industry (Total: {total_firms})", pad=12)

    ax.tick_params(axis="y", colors="0.25")

    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="-", linewidth=0.8, alpha=0.18, color="0.6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = max(antall) if antall else 0
    ax.set_xlim(0, x_max * 1.18 + 0.5)

    for b in bars:
        w = b.get_width()
        ax.text(
            w + (x_max * 0.02 + 0.05),
            b.get_y() + b.get_height() / 2,
            f"{int(w)}",
            va="center",
            ha="left",
            color="0.2",
            fontsize=11
        )

    fig.tight_layout()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.show()


def lag_piechart(sektor_teller, out_plot: Path):
    """
    Lager sektordiagram for sektorfordeling med blåtoner.
    """
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    sorterte = sorted(sektor_teller.items(), key=lambda x: x[1], reverse=True)
    sektorer = [x[0] for x in sorterte]
    antall = [x[1] for x in sorterte]

    # Basefarge: dus/tåkete navy
    base_color = np.array(mcolors.to_rgb("#5f6d85"))

    # Lager lysere blåtoner av samme base
    colors = []
    for i in np.linspace(0.12, 0.72, len(sektorer)):
        new_color = base_color + (1 - base_color) * i
        colors.append(new_color)

    plt.figure(figsize=(8, 8))
    plt.pie(
        antall,
        labels=sektorer,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    plt.title("Sector distribution")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    base_dir = Path(__file__).resolve().parent

    excel_filer = [
        f for f in base_dir.rglob("*.xlsx")
        if not f.name.startswith("~$")
    ]

    if not excel_filer:
        print("Fant ingen .xlsx-filer.")
        return

    selskaper_og_sektorer = []
    mangler_sektor = []
    unmapped_sektorer = set()

    for fil in excel_filer:
        sektor_raw = hent_sektor_fra_fil(fil)
        selskap = fil.stem

        if sektor_raw:
            sektor_mapped = map_sektor(sektor_raw)
            selskaper_og_sektorer.append((selskap, sektor_mapped))

            if (
                sektor_raw not in SECTOR_GROUP_MAP
                and "pharma" not in sektor_raw.lower()
                and "biotech" not in sektor_raw.lower()
                and "healthcare" not in sektor_raw.lower()
                and "health care" not in sektor_raw.lower()
            ):
                unmapped_sektorer.add(sektor_raw)
        else:
            mangler_sektor.append(selskap)

    if not selskaper_og_sektorer:
        print("Fant ingen sektorer i celle B5.")
        return

    sektor_teller = Counter(sektor for _, sektor in selskaper_og_sektorer)

    sorterte_desc = sorted(sektor_teller.items(), key=lambda x: x[1], reverse=True)

    print("\nSelskaper og grupperte sektorer:")
    for selskap, sektor in sorted(selskaper_og_sektorer):
        print(f"{selskap}: {sektor}")

    print("\nFordeling per gruppert sektor:")
    for sektor, count in sorterte_desc:
        print(f"{sektor}: {count}")

    if unmapped_sektorer:
        print("\nSektorer som ikke var i SECTOR_GROUP_MAP:")
        for sektor in sorted(unmapped_sektorer):
            print(f"- {sektor}")

    if mangler_sektor:
        print("\nFiler uten gyldig sektor i B5:")
        for navn in mangler_sektor:
            print(f"- {navn}")

    lag_barplot(sektor_teller, base_dir / "sektorfordeling_barh.png")
    lag_piechart(sektor_teller, base_dir / "sektorfordeling_pie.png")


if __name__ == "__main__":
    main()