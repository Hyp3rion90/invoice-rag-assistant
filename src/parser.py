"""
parser.py
Turn a FatturaPA 1.2 (SDI) XML invoice into a clean Python dictionary.

This is Fase 1 of the Invoice RAG Assistant: before we can embed, index or
query anything, we must extract structured data from the raw XML.

Run it directly to parse every file in data/raw_xml/ and preview the result:
    python src/parser.py
"""

from pathlib import Path
from pprint import pprint
from lxml import etree

# ---------------------------------------------------------------------------
# The namespace — the single most important concept in this file.
# ---------------------------------------------------------------------------
# The root element declares:  xmlns="http://ivaservizi.agenziaentrate.gov.it/docs/xsd/fatture/v1.2"
# That is a DEFAULT namespace: every tag in the document silently belongs to it.
# So the parser does NOT see a tag called "Descrizione" — it sees a tag whose
# full name is "{http://...v1.2}Descrizione".
# If you search for the bare name you get None. We must always qualify it.
NS_URI = "http://ivaservizi.agenziaentrate.gov.it/docs/xsd/fatture/v1.2"

# Human-readable mapping for SDI payment codes (extend as needed).
PAYMENT_CODES = {
    "MP01": "contanti",
    "MP05": "bonifico",
    "MP08": "carta",
}


def _tag(name):
    """Qualify a tag name with the SDI namespace, in 'Clark notation'.
    'Descrizione' -> '{http://...v1.2}Descrizione'"""
    return f"{{{NS_URI}}}{name}"


def _text(element, *path, default=None):
    """Walk down a chain of child tags and return the text of the last one.

    Example:
        _text(cedente, "DatiAnagrafici", "Anagrafica", "Denominazione")
    Returns `default` if any step in the chain is missing, so a slightly
    different invoice never crashes the parser — it just yields None.
    """
    node = element
    for name in path:
        if node is None:
            return default
        node = node.find(_tag(name))
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _to_float(value, default=0.0):
    """Safely convert SDI numeric text to float (decimal separator is '.')."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_invoice(path):
    """Parse a single FatturaPA XML file into a structured dict."""
    tree = etree.parse(str(path))
    root = tree.getroot()

    header = root.find(_tag("FatturaElettronicaHeader"))
    body = root.find(_tag("FatturaElettronicaBody"))

    # --- parties (supplier / customer) -----------------------------------
    cedente = header.find(_tag("CedentePrestatore"))
    cessionario = header.find(_tag("CessionarioCommittente"))

    fornitore = _text(cedente, "DatiAnagrafici", "Anagrafica", "Denominazione")
    piva_fornitore = "{}{}".format(
        _text(cedente, "DatiAnagrafici", "IdFiscaleIVA", "IdPaese", default=""),
        _text(cedente, "DatiAnagrafici", "IdFiscaleIVA", "IdCodice", default=""),
    )
    cliente = _text(cessionario, "DatiAnagrafici", "Anagrafica", "Denominazione")
    piva_cliente = "{}{}".format(
        _text(cessionario, "DatiAnagrafici", "IdFiscaleIVA", "IdPaese", default=""),
        _text(cessionario, "DatiAnagrafici", "IdFiscaleIVA", "IdCodice", default=""),
    )

    # --- general document data -------------------------------------------
    generali = body.find(_tag("DatiGenerali")).find(_tag("DatiGeneraliDocumento"))
    numero = _text(generali, "Numero")
    # NOTE: we keep the date in ISO format (YYYY-MM-DD) on purpose. ISO strings
    # sort and filter correctly (essential for the "maggio 2026" query later).
    # We format to DD/MM/YYYY only when showing it to a human.
    data = _text(generali, "Data")

    # --- line items ------------------------------------------------------
    dati_beni = body.find(_tag("DatiBeniServizi"))
    righe = []
    for linea in dati_beni.findall(_tag("DettaglioLinee")):
        has_sconto = linea.find(_tag("ScontoMaggiorazione")) is not None
        righe.append({
            "descrizione": _text(linea, "Descrizione"),
            "importo": _to_float(_text(linea, "PrezzoTotale")),
            "iva": _to_float(_text(linea, "AliquotaIVA")),
            "natura_iva": _text(linea, "Natura"),   # None when standard VAT
            "sconto": has_sconto,
        })

    # --- VAT summary (DatiRiepilogo) -> totals ---------------------------
    riepiloghi = []
    for r in dati_beni.findall(_tag("DatiRiepilogo")):
        riepiloghi.append({
            "aliquota": _to_float(_text(r, "AliquotaIVA")),
            "natura_iva": _text(r, "Natura"),
            "imponibile": _to_float(_text(r, "ImponibileImporto")),
            "imposta": _to_float(_text(r, "Imposta")),
        })
    imponibile = round(sum(x["imponibile"] for x in riepiloghi), 2)
    iva_totale = round(sum(x["imposta"] for x in riepiloghi), 2)

    # Prefer the declared document total; fall back to computed one.
    totale = _to_float(_text(generali, "ImportoTotaleDocumento"),
                       default=round(imponibile + iva_totale, 2))

    # --- payment ---------------------------------------------------------
    pagamento_node = body.find(_tag("DatiPagamento"))
    modalita = _text(pagamento_node, "DettaglioPagamento", "ModalitaPagamento")
    data_scadenza = _text(pagamento_node, "DettaglioPagamento", "DataScadenzaPagamento")

    # --- fiscal regime (domain logic) ------------------------------------
    # Natura codes N6.x = "inversione contabile" (reverse charge).
    # N6.6 specifically = reverse charge on electronics (art. 17 c.6 lett.c).
    nature = [r["natura_iva"] for r in righe if r["natura_iva"]]
    if any(n.startswith("N6") for n in nature):
        regime = "reverse_charge"
    elif nature:
        regime = "altro_regime"   # e.g. N4 esente, N2 non soggetto, ...
    else:
        regime = "ordinario"

    return {
        "file": Path(path).name,
        "numero_fattura": numero,
        "data": data,                       # ISO: "2026-05-28"
        "fornitore": fornitore,
        "piva_fornitore": piva_fornitore,
        "cliente": cliente,
        "piva_cliente": piva_cliente,
        "righe": righe,
        "imponibile": imponibile,
        "iva_totale": iva_totale,
        "totale": totale,
        "modalita_pagamento": PAYMENT_CODES.get(modalita, modalita),
        "data_scadenza": data_scadenza,
        "ha_sconto": any(r["sconto"] for r in righe),
        "regime": regime,
    }


def parse_all(folder="data/raw_xml"):
    """Parse every .xml file in a folder. Returns a list of dicts.
    Files that fail are reported but do not stop the whole run."""
    invoices = []
    for path in sorted(Path(folder).glob("*.xml")):
        try:
            invoices.append(parse_invoice(path))
        except Exception as e:
            print(f"[SKIP] {path.name}: {type(e).__name__} - {e}")
    return invoices


if __name__ == "__main__":
    data = parse_all()
    print(f"Parsed {len(data)} invoices.\n")
    if data:
        print("Preview of the first parsed invoice:")
        pprint(data[0], sort_dicts=False)
