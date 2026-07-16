"""
vectorstore.py
Fase 2: build a semantic index of the invoice line items in ChromaDB.

Design: each invoice LINE becomes one chunk. A chunk has two parts:
  - a short natural-language sentence  -> used for SEMANTIC search
  - a metadata dict (typed values)     -> used for EXACT filtering (where=...)

Run it to (re)build the index and try example searches:
    python src/vectorstore.py     (run from the project root)
"""

import chromadb
from chromadb.utils import embedding_functions

from parser import parse_all   # our Fase 1 parser

# --- configuration --------------------------------------------------------
DB_PATH = "chroma_db"                 # on-disk store (already in .gitignore)
COLLECTION_NAME = "fatture"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # free, multilingual, CPU-ok


def build_chunk_text(invoice, riga):
    """Turn one line item into a short Italian sentence for semantic search.
    We enrich it with domain phrases (e.g. 'reverse charge') so the meaning is
    searchable, not just the raw code 'N6.6'."""
    parts = [
        f"Fattura {invoice['numero_fattura']} del {invoice['data']}",
        f"fornitore {invoice['fornitore']}",
        f"descrizione: {riga['descrizione']}",
        f"importo {riga['importo']:.2f} euro",
        f"aliquota IVA {riga['iva']:.0f}%",
    ]
    if riga["natura_iva"]:
        parts.append(f"natura IVA {riga['natura_iva']}")
        if riga["natura_iva"].startswith("N6"):
            parts.append("reverse charge inversione contabile")
    if riga["sconto"]:
        parts.append("con sconto applicato")
    parts.append(f"pagamento {invoice['modalita_pagamento']}")
    return " - ".join(parts)


def build_metadata(invoice, riga):
    """Structured metadata for exact filtering.
    IMPORTANT: ChromaDB metadata values must be str / int / float / bool.
    None is NOT allowed, so we convert missing natura to a placeholder."""
    return {
        "numero_fattura": invoice["numero_fattura"],
        "fornitore": invoice["fornitore"],
        "data": invoice["data"],              # ISO "2026-05-25"
        "mese": invoice["data"][:7],          # "2026-05" -> easy month filter
        "natura_iva": riga["natura_iva"] or "nessuna",
        "aliquota": riga["iva"],              # float, e.g. 22.0
        "importo": riga["importo"],           # float
        "modalita_pagamento": invoice["modalita_pagamento"],
        "regime": invoice["regime"],
        "ha_sconto": riga["sconto"],          # bool
    }


def build_index():
    """Parse all invoices and (re)build the ChromaDB collection from scratch."""
    invoices = parse_all()
    if not invoices:
        print("No invoices found in data/raw_xml/. Run generate_invoices.py first.")
        return None

    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild from scratch each run so re-running never creates duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet — fine

    # The embedding function turns text into vectors automatically, both when
    # we add documents and when we later query. First run downloads the model
    # (~470 MB) — that is normal and happens only once.
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

    documents, metadatas, ids = [], [], []
    for inv in invoices:
        for i, riga in enumerate(inv["righe"]):
            documents.append(build_chunk_text(inv, riga))
            metadatas.append(build_metadata(inv, riga))
            ids.append(f"{inv['file']}#{i}")   # stable unique id per line

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(documents)} line-chunks from {len(invoices)} invoices.\n")
    return collection


def print_results(title, docs, metas, distances=None):
    """Pretty-print a handful of search results."""
    print(f"=== {title} ===")
    if not docs:
        print("  (no results)\n")
        return
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        dist = f"  [dist={distances[i]:.3f}]" if distances else ""
        print(f"  {meta['numero_fattura']} | {meta['fornitore']}{dist}")
        print(f"    {doc}")
    print()


if __name__ == "__main__":
    collection = build_index()
    if collection is None:
        raise SystemExit

    # --- TEST 1: pure SEMANTIC search (Famiglia A) ------------------------
    # We never wrote the words "reverse charge" in the query's exact chunks by
    # code, yet the meaning matches N6.6 lines.
    res = collection.query(query_texts=["fatture in reverse charge"], n_results=5)
    print_results("Semantic query: 'fatture in reverse charge'",
                  res["documents"][0], res["metadatas"][0], res["distances"][0])

    # --- TEST 2: pure METADATA filter, no semantics (Famiglia A) ----------
    # collection.get() with a where filter returns ALL matches, not top-K.
    got = collection.get(where={"mese": "2026-05"})
    print(f"=== Metadata filter: mese == '2026-05' ===")
    print(f"  {len(got['ids'])} line-chunks are dated May 2026.\n")

    # --- TEST 3: exact aliquota filter ------------------------------------
    got22 = collection.get(where={"aliquota": 22.0})
    print(f"=== Metadata filter: aliquota == 22% ===")
    print(f"  {len(got22['ids'])} line-chunks have 22% VAT.\n")

    # --- TEST 4: SEMANTIC + FILTER combined -------------------------------
    res2 = collection.query(
        query_texts=["dispositivi di rete"],
        n_results=3,
        where={"regime": "reverse_charge"},
    )
    print_results("Semantic 'dispositivi di rete' filtered to reverse_charge",
                  res2["documents"][0], res2["metadatas"][0], res2["distances"][0])
