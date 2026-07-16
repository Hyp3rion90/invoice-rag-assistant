"""
analytics.py
Fase 3 (part 1): the DETERMINISTIC half of the hybrid system.

Vector search is great at "find lines that MEAN X", but wrong at
"SUM / MAX / GROUP BY", because it returns the top-K most similar chunks,
not ALL matches. So we load every invoice into pandas and answer the numeric
questions exactly. Claude will sit on top of this later; keeping the maths
here means Claude can never invent a number.

    python src/analytics.py     (run from the project root)
"""

import pandas as pd
from parser import parse_all


def load_dataframes(folder="data/raw_xml"):
    """Build two tables from the parsed invoices:
       - inv_df : one row per INVOICE  (for totals, payment, regime)
       - line_df: one row per LINE     (for VAT nature, descriptions)
    """
    invoices = parse_all(folder)

    inv_rows, line_rows = [], []
    for inv in invoices:
        inv_rows.append({
            "numero": inv["numero_fattura"],
            "data": inv["data"],
            "mese": inv["data"][:7],                 # "2026-05"
            "fornitore": inv["fornitore"],
            "imponibile": inv["imponibile"],
            "iva_totale": inv["iva_totale"],
            "totale": inv["totale"],
            "regime": inv["regime"],                 # invoice-level
            "pagamento": inv["modalita_pagamento"],
            "ha_sconto": inv["ha_sconto"],
        })
        for r in inv["righe"]:
            line_rows.append({
                "numero": inv["numero_fattura"],
                "mese": inv["data"][:7],
                "fornitore": inv["fornitore"],
                "descrizione": r["descrizione"],
                "importo": r["importo"],
                "iva": r["iva"],
                "natura_iva": r["natura_iva"] or "nessuna",   # line-level
                "sconto": r["sconto"],
            })

    return pd.DataFrame(inv_rows), pd.DataFrame(line_rows)


# --- reusable analytics functions -----------------------------------------
def totale_per_fornitore(inv_df):
    """Total invoiced amount grouped by supplier, biggest first."""
    return (inv_df.groupby("fornitore")["totale"]
            .sum()
            .sort_values(ascending=False))


def riepilogo_per_natura(line_df):
    """For each VAT nature: how many lines and total taxable amount."""
    return (line_df.groupby("natura_iva")
            .agg(righe=("importo", "size"),
                 imponibile=("importo", "sum"))
            .sort_values("imponibile", ascending=False))


# --- demo: answer all 8 test questions from the brief ---------------------
if __name__ == "__main__":
    inv_df, line_df = load_dataframes()
    print(f"{len(inv_df)} fatture, {len(line_df)} righe.\n")

    # Q1 — which invoices are reverse charge? (note: Amazon shows up here,
    #      the exact thing the semantic top-5 had MISSED)
    rc = inv_df[inv_df["regime"] == "reverse_charge"]
    print(f"Q1 Fatture reverse charge: {len(rc)} | "
          f"fornitori: {sorted(rc['fornitore'].unique())}")

    # Q2 — total spent with Amazon (a SUM: vector search would get this wrong)
    amazon = inv_df[inv_df["fornitore"].str.contains("Amazon")]["totale"].sum()
    print(f"Q2 Totale speso con Amazon: {amazon:.2f} EUR")

    # Q3 — invoices dated May 2026
    print(f"Q3 Fatture di maggio 2026: {len(inv_df[inv_df['mese'] == '2026-05'])}")

    # Q4 — lines with 22% VAT
    print(f"Q4 Righe con IVA 22%: {len(line_df[line_df['iva'] == 22.0])}")

    # Q5 — supplier with the highest cumulative amount (GROUP BY + MAX)
    per_forn = totale_per_fornitore(inv_df)
    print(f"Q5 Fornitore con importo piu alto: {per_forn.index[0]} "
          f"({per_forn.iloc[0]:.2f} EUR)")

    # Q6 — invoices paid by card
    print(f"Q6 Fatture pagate con carta: {len(inv_df[inv_df['pagamento'] == 'carta'])}")

    # Q7 — summary by VAT nature (GROUP BY)
    print("\nQ7 Riepilogo per natura IVA:")
    print(riepilogo_per_natura(line_df).to_string())

    # Q8 — invoices with a discount
    print(f"\nQ8 Fatture con sconto: {len(inv_df[inv_df['ha_sconto']])}")
