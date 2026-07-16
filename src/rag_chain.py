"""
rag_chain.py
Fase 3-4: the hybrid RAG pipeline, now CONVERSATION-AWARE.

Data goes in the `system` prompt (stable within a turn); the conversation
history + the current question go in `messages`. That gives Claude memory of
previous turns, so follow-up questions ("e di queste, solo a maggio?") work.

    python src/rag_chain.py     (batch test, no memory)

Requires:
  - ChromaDB index already built (run vectorstore.py once)
  - ANTHROPIC_API_KEY in a .env file in the project root
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from analytics import load_dataframes
from vectorstore import DB_PATH, COLLECTION_NAME, EMBED_MODEL

load_dotenv()
client = Anthropic()

# temperature=0 works on Haiku 4.5. On "claude-sonnet-5" you MUST remove it (400).
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_INSTRUCTIONS = """Sei un assistente contabile esperto di fatturazione elettronica italiana.
Rispondi in italiano in modo preciso e conciso.
Basa la risposta SOLO sui DATI qui sotto e sulle RIGHE RILEVANTI nel messaggio dell'utente.
Per totali e conteggi usa il RIEPILOGO DETERMINISTICO: sono numeri esatti, non ricalcolarli.
Per elencare TUTTE le fatture che soddisfano un criterio usa l'ELENCO COMPLETO FATTURE.
Per gli elenchi usa un formato compatto: una fattura per riga (numero, fornitore, totale), senza titoli markdown superflui.
Tieni conto della conversazione precedente: se l'utente fa una domanda di follow-up
(es. "e di queste?", "solo a maggio?"), interpretala alla luce delle risposte gia' date.
Cita sempre numero e fornitore delle fatture. Se un'informazione non e' presente, dillo esplicitamente."""


def get_collection():
    """Connect to the already-built ChromaDB collection (does NOT rebuild it)."""
    chroma = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    try:
        return chroma.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        raise SystemExit("Index not found. Run 'python src/vectorstore.py' first.")


def deterministic_summary(inv_df, line_df):
    """Compact, EXACT snapshot of the dataset, computed with pandas."""
    parts = ["RIEPILOGO DETERMINISTICO (calcolato con pandas, numeri esatti):"]

    per_forn = (inv_df.groupby("fornitore")
                .agg(fatture=("numero", "size"), totale=("totale", "sum"))
                .sort_values("totale", ascending=False))
    parts.append("\nTotale e numero fatture per fornitore:")
    for forn, row in per_forn.iterrows():
        parts.append(f"- {forn}: {row['totale']:.2f} EUR su {int(row['fatture'])} fatture")

    per_nat = (line_df.groupby("natura_iva")
               .agg(righe=("importo", "size"), imponibile=("importo", "sum"))
               .sort_values("imponibile", ascending=False))
    parts.append("\nRiepilogo per natura IVA:")
    for nat, row in per_nat.iterrows():
        parts.append(f"- {nat}: {int(row['righe'])} righe, {row['imponibile']:.2f} EUR imponibile")

    per_aliq = (line_df.groupby("iva")
                .agg(righe=("importo", "size"), imponibile=("importo", "sum"))
                .sort_index(ascending=False))
    parts.append("\nRighe per aliquota IVA:")
    for aliq, row in per_aliq.iterrows():
        parts.append(f"- {aliq:.0f}%: {int(row['righe'])} righe, {row['imponibile']:.2f} EUR imponibile")

    per_mese = inv_df.groupby("mese").size().sort_index()
    parts.append("\nFatture per mese: " +
                 ", ".join(f"{m}={int(c)}" for m, c in per_mese.items()))

    parts.append(
        f"\nConteggi: fatture totali={len(inv_df)}, "
        f"pagate con carta={int((inv_df['pagamento'] == 'carta').sum())}, "
        f"con sconto={int(inv_df['ha_sconto'].sum())}, "
        f"in reverse charge={int((inv_df['regime'] == 'reverse_charge').sum())}"
    )
    return "\n".join(parts)


def elenco_fatture(inv_df, line_df):
    """One compact line per invoice, with the VAT rates and natures it contains."""
    aliq_map = line_df.groupby("numero")["iva"].apply(lambda s: sorted(set(s)))
    nat_map = (line_df.groupby("numero")["natura_iva"]
               .apply(lambda s: sorted({v for v in s if v != "nessuna"})))

    rows = ["ELENCO COMPLETO FATTURE (tutte le fatture; usa questo per elenchi e filtri):"]
    for _, inv in inv_df.sort_values("data").iterrows():
        num = inv["numero"]
        aliquote = ", ".join(f"{a:.0f}%" for a in aliq_map.get(num, []))
        nature = ", ".join(nat_map.get(num, [])) or "-"
        sconto = "sconto" if inv["ha_sconto"] else "no-sconto"
        rows.append(
            f"- {num} | {inv['data']} | {inv['fornitore']} | tot {inv['totale']:.2f} EUR | "
            f"{inv['regime']} | pagamento {inv['pagamento']} | {sconto} | "
            f"aliquote: {aliquote} | nature: {nature}"
        )
    return "\n".join(rows)


def build_data_context(inv_df, line_df):
    """The stable, dataset-level context that goes into the system prompt."""
    return f"{deterministic_summary(inv_df, line_df)}\n\n{elenco_fatture(inv_df, line_df)}"


def retrieve_lines(question, collection, n=6):
    """Semantic retrieval: the most relevant invoice lines for THIS question."""
    res = collection.query(query_texts=[question], n_results=n)
    return res["documents"][0]


def answer(question, history, inv_df, line_df, collection, n_semantic=6):
    """Conversation-aware pipeline.
    `history` is a list of prior turns: [{"role": "user"/"assistant", "content": ...}].
    Returns (testo_risposta, righe_usate) so the UI can show the sources.
    """
    system = f"{SYSTEM_INSTRUCTIONS}\n\nDATI:\n{build_data_context(inv_df, line_df)}"
    righe = retrieve_lines(question, collection, n_semantic)
    righe_txt = "\n".join(f"- {c}" for c in righe)
    user_content = f"{question}\n\nRIGHE RILEVANTI (per le descrizioni prodotto):\n{righe_txt}"

    messages = list(history) + [{"role": "user", "content": user_content}]
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,          # raised from 1000 so long lists aren't truncated
        temperature=0,
        system=system,
        messages=messages,
    )
    return msg.content[0].text, righe


if __name__ == "__main__":
    inv_df, line_df = load_dataframes()
    collection = get_collection()

    domande = [
        "Quali fatture sono soggette a reverse charge?",
        "Qual e' il totale speso con Amazon Business?",
        "Mostrami tutte le fatture del mese di maggio 2026",
        "Ci sono fatture con IVA al 22%?",
        "Qual e' il fornitore con l'importo piu' alto?",
        "Quali fatture hanno metodo di pagamento con carta?",
        "Dammi un riepilogo delle fatture per natura IVA",
        "C'e' qualche fattura con sconto applicato?",
    ]
    for q in domande:
        text, _ = answer(q, [], inv_df, line_df, collection)   # [] = no memory in batch
        print("=" * 72)
        print("D:", q)
        print("R:", text)
        print()