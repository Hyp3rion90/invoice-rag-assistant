"""
generate_invoices.py
Synthetic Italian e-invoice (FatturaPA 1.2) generator.

Why this file exists:
- The RAG project needs realistic SDI XML invoices to work on.
- Real company invoices must NEVER be committed to a public repo (privacy).
- So we generate fake-but-realistic invoices, modeled on the official format.

IMPORTANT: the output looks like a FatturaPA but is for DEVELOPMENT ONLY.
It is not certified/valid for real submission to the Sistema di Interscambio.

Usage (run from the repo root):
    python src/generate_invoices.py

Output:
    data/raw_xml/IT<piva>_<progressivo>.xml   (one file per invoice)
"""

import os
import random
import datetime
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join("data", "raw_xml")
N_INVOICES = 50
SEED = 42  # fixed seed -> the dataset is reproducible (same files every run)

random.seed(SEED)

# Official SDI namespace for FatturaPA 1.2. The parser will have to be
# namespace-aware later, exactly like with real invoices.
NAMESPACE = "http://ivaservizi.agenziaentrate.gov.it/docs/xsd/fatture/v1.2"

# Fixed buyer (CessionarioCommittente). Fictional ON PURPOSE: never put a real
# company name / VAT number on a public repo.
BUYER = {
    "denominazione": "Innovatech Solutions Srl",
    "piva": "IT12345670018",
    "indirizzo": "Via Garibaldi 10",
    "cap": "20121",
    "comune": "Milano",
    "provincia": "MI",
}

# Suppliers (CedentePrestatore). Each has its OWN small catalog so that the
# VAT profiles stay realistic (an electronics seller uses reverse charge N6.6,
# a consultant uses 22%, a hotel uses 10%, etc.).
#
# catalog item = (description, unit_price, vat_rate, nature_code_or_None)
#   vat_rate == 0.0 together with a "Natura" code => exempt / reverse charge.
SUPPLIERS = [
    {
        "denominazione": "Amazon Business EU S.a.r.l",
        "piva": "IT13397910962",
        "indirizzo": "Viale Monte Grappa 3",
        "cap": "20124",
        "comune": "Milano",
        "provincia": "MI",
        "catalog": [
            ("Samsung Galaxy Tab S10 Ultra", 719.67, 0.0, "N6.6"),
            ("Notebook Dell Latitude 5540", 1180.00, 0.0, "N6.6"),
            ("Monitor LG UltraWide 34 pollici", 410.00, 0.0, "N6.6"),
            ("SSD Samsung 990 PRO 2TB", 189.90, 0.0, "N6.6"),
        ],
    },
    {
        "denominazione": "Tecno Forniture Srl",
        "piva": "IT01234560017",
        "indirizzo": "Corso Francia 120",
        "cap": "10143",
        "comune": "Torino",
        "provincia": "TO",
        "catalog": [
            ("Switch di rete gestito 24 porte", 240.00, 0.0, "N6.6"),
            ("Gruppo di continuita UPS 1500VA", 320.00, 0.0, "N6.6"),
            ("Cavi HDMI confezione 10 pezzi", 49.00, 22.0, None),
            ("Tastiera e mouse wireless", 38.50, 22.0, None),
        ],
    },
    {
        "denominazione": "Ufficio Piu Spa",
        "piva": "IT09876540015",
        "indirizzo": "Via Tortona 25",
        "cap": "20144",
        "comune": "Milano",
        "provincia": "MI",
        "catalog": [
            ("Risme carta A4 (cartone da 5)", 32.50, 22.0, None),
            ("Toner compatibile HP", 96.00, 22.0, None),
            ("Cancelleria varia ufficio", 85.50, 22.0, None),
        ],
    },
    {
        "denominazione": "Consulenze Rossi e Associati",
        "piva": "IT05554440018",
        "indirizzo": "Via Nazionale 50",
        "cap": "00184",
        "comune": "Roma",
        "provincia": "RM",
        "catalog": [
            ("Consulenza fiscale (8 ore)", 600.00, 22.0, None),
            ("Redazione bilancio annuale", 1500.00, 22.0, None),
            ("Prestazione esente art.10 DPR 633/72", 200.00, 0.0, "N4"),
        ],
    },
    {
        "denominazione": "Hotel Centrale Srl",
        "piva": "IT04443330016",
        "indirizzo": "Via Dante 8",
        "cap": "20121",
        "comune": "Milano",
        "provincia": "MI",
        "catalog": [
            ("Pernottamento camera doppia", 120.00, 10.0, None),
            ("Pranzo di lavoro", 45.00, 10.0, None),
            ("Servizio sala riunioni mezza giornata", 250.00, 22.0, None),
        ],
    },
    {
        "denominazione": "Energia Futura Spa",
        "piva": "IT07778880011",
        "indirizzo": "Via Stalingrado 45",
        "cap": "40128",
        "comune": "Bologna",
        "provincia": "BO",
        "catalog": [
            ("Fornitura energia elettrica - mese", 430.00, 22.0, None),
            ("Fornitura gas - mese", 210.00, 10.0, None),
        ],
    },
]

# Payment methods (SDI codes): MP05=bonifico, MP08=carta di credito, MP01=contanti
PAYMENT_METHODS = ["MP05", "MP05", "MP08", "MP08", "MP01"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_invoice_date():
    """Return a date. ~35% of invoices fall in May 2026 so that the
    'maggio 2026' test query has plenty of matches."""
    if random.random() < 0.35:
        return datetime.date(2026, 5, random.randint(1, 28))
    start = datetime.date(2025, 6, 1)
    end = datetime.date(2026, 6, 30)
    span = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, span))


def to_base36(n, width=5):
    """Encode an int as a fixed-width base36 string (SDI 'progressivo' style)."""
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    s = ""
    n = max(n, 1)
    while n:
        n, r = divmod(n, 36)
        s = chars[r] + s
    return s.rjust(width, "0")


def build_line(num, item):
    """Build one <DettaglioLinee> block.
    Returns (xml_string, vat_rate, nature, line_total)."""
    desc, unit_price, vat, nature = item
    qty = random.choice([1, 1, 1, 2, 3])
    gross = unit_price * qty

    # 20% chance this line carries a 10% discount (covers the 'sconto' query)
    sconto_xml = ""
    discount = 0.0
    if random.random() < 0.20:
        discount = round(gross * 0.10, 2)
        sconto_xml = (
            "      <ScontoMaggiorazione>\n"
            "        <Tipo>SC</Tipo>\n"
            "        <Percentuale>10.00</Percentuale>\n"
            f"        <Importo>{discount:.2f}</Importo>\n"
            "      </ScontoMaggiorazione>\n"
        )

    total = round(gross - discount, 2)
    natura_xml = f"      <Natura>{nature}</Natura>\n" if nature else ""

    line_xml = (
        "    <DettaglioLinee>\n"
        f"      <NumeroLinea>{num}</NumeroLinea>\n"
        f"      <Descrizione>{escape(desc)}</Descrizione>\n"
        f"      <Quantita>{qty:.2f}</Quantita>\n"
        f"      <PrezzoUnitario>{unit_price:.2f}</PrezzoUnitario>\n"
        f"{sconto_xml}"
        f"      <PrezzoTotale>{total:.2f}</PrezzoTotale>\n"
        f"      <AliquotaIVA>{vat:.2f}</AliquotaIVA>\n"
        f"{natura_xml}"
        "    </DettaglioLinee>\n"
    )
    return line_xml, vat, nature, total


def party_block(tag, party, include_regime=False):
    """Build a CedentePrestatore / CessionarioCommittente block."""
    code = party["piva"][2:]  # strip the 'IT' prefix for IdCodice
    regime = "        <RegimeFiscale>RF01</RegimeFiscale>\n" if include_regime else ""
    return (
        f"  <{tag}>\n"
        "    <DatiAnagrafici>\n"
        "      <IdFiscaleIVA>\n"
        "        <IdPaese>IT</IdPaese>\n"
        f"        <IdCodice>{code}</IdCodice>\n"
        "      </IdFiscaleIVA>\n"
        "      <Anagrafica>\n"
        f"        <Denominazione>{escape(party['denominazione'])}</Denominazione>\n"
        "      </Anagrafica>\n"
        f"{regime}"
        "    </DatiAnagrafici>\n"
        "    <Sede>\n"
        f"      <Indirizzo>{escape(party['indirizzo'])}</Indirizzo>\n"
        f"      <CAP>{party['cap']}</CAP>\n"
        f"      <Comune>{escape(party['comune'])}</Comune>\n"
        f"      <Provincia>{party['provincia']}</Provincia>\n"
        "      <Nazione>IT</Nazione>\n"
        "    </Sede>\n"
        f"  </{tag}>\n"
    )


def build_invoice(seq):
    """Build one full invoice. Returns (filename, xml_string)."""
    supplier = random.choice(SUPPLIERS)
    date = random_invoice_date()
    progressivo = to_base36(seq)
    numero = f"{date.year}/{seq:04d}"
    payment = random.choice(PAYMENT_METHODS)

    # --- build 1..3 lines -------------------------------------------------
    n_lines = random.randint(1, 3)
    lines_xml = ""
    line_meta = []  # list of (total, vat, nature)
    for i in range(1, n_lines + 1):
        item = random.choice(supplier["catalog"])
        xml, vat, nature, total = build_line(i, item)
        lines_xml += xml
        line_meta.append((total, vat, nature))

    # --- group by (vat, nature) for DatiRiepilogo -------------------------
    grouped = {}
    for total, vat, nature in line_meta:
        key = (vat, nature)
        grouped[key] = round(grouped.get(key, 0.0) + total, 2)

    riepilogo_xml = ""
    imposta_total = 0.0
    for (vat, nature), imponibile in sorted(grouped.items(),
                                            key=lambda x: (-x[0][0], str(x[0][1]))):
        imposta = round(imponibile * vat / 100.0, 2)
        imposta_total = round(imposta_total + imposta, 2)
        natura_line = f"      <Natura>{nature}</Natura>\n" if nature else ""
        esig_line = "      <EsigibilitaIVA>I</EsigibilitaIVA>\n" if not nature else ""
        riepilogo_xml += (
            "    <DatiRiepilogo>\n"
            f"      <AliquotaIVA>{vat:.2f}</AliquotaIVA>\n"
            f"{natura_line}"
            f"      <ImponibileImporto>{imponibile:.2f}</ImponibileImporto>\n"
            f"      <Imposta>{imposta:.2f}</Imposta>\n"
            f"{esig_line}"
            "    </DatiRiepilogo>\n"
        )

    imponibile_total = round(sum(grouped.values()), 2)
    doc_total = round(imponibile_total + imposta_total, 2)
    due_date = date + datetime.timedelta(days=30)

    # --- header -----------------------------------------------------------
    trasmittente_code = supplier["piva"][2:]
    header = (
        "  <FatturaElettronicaHeader>\n"
        "    <DatiTrasmissione>\n"
        "      <IdTrasmittente>\n"
        "        <IdPaese>IT</IdPaese>\n"
        f"        <IdCodice>{trasmittente_code}</IdCodice>\n"
        "      </IdTrasmittente>\n"
        f"      <ProgressivoInvio>{progressivo}</ProgressivoInvio>\n"
        "      <FormatoTrasmissione>FPR12</FormatoTrasmissione>\n"
        "      <CodiceDestinatario>0000000</CodiceDestinatario>\n"
        "    </DatiTrasmissione>\n"
        + party_block("CedentePrestatore", supplier, include_regime=True)
        + party_block("CessionarioCommittente", BUYER)
        + "  </FatturaElettronicaHeader>\n"
    )

    # --- body -------------------------------------------------------------
    body = (
        "  <FatturaElettronicaBody>\n"
        "    <DatiGenerali>\n"
        "      <DatiGeneraliDocumento>\n"
        "        <TipoDocumento>TD01</TipoDocumento>\n"
        "        <Divisa>EUR</Divisa>\n"
        f"        <Data>{date.isoformat()}</Data>\n"
        f"        <Numero>{numero}</Numero>\n"
        f"        <ImportoTotaleDocumento>{doc_total:.2f}</ImportoTotaleDocumento>\n"
        "      </DatiGeneraliDocumento>\n"
        "    </DatiGenerali>\n"
        "    <DatiBeniServizi>\n"
        f"{lines_xml}"
        f"{riepilogo_xml}"
        "    </DatiBeniServizi>\n"
        "    <DatiPagamento>\n"
        "      <CondizioniPagamento>TP02</CondizioniPagamento>\n"
        "      <DettaglioPagamento>\n"
        f"        <ModalitaPagamento>{payment}</ModalitaPagamento>\n"
        f"        <DataScadenzaPagamento>{due_date.isoformat()}</DataScadenzaPagamento>\n"
        f"        <ImportoPagamento>{doc_total:.2f}</ImportoPagamento>\n"
        "      </DettaglioPagamento>\n"
        "    </DatiPagamento>\n"
        "  </FatturaElettronicaBody>\n"
    )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<FatturaElettronica versione="FPR12" xmlns="{NAMESPACE}">\n'
        + header
        + body
        + "</FatturaElettronica>\n"
    )

    filename = f"{supplier['piva']}_{progressivo}.xml"
    return filename, xml


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for seq in range(1, N_INVOICES + 1):
        filename, xml = build_invoice(seq)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)
    print(f"Done. {N_INVOICES} invoices written to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
