"""
app.py
Fase 4-5: the Streamlit interface for the Invoice RAG Assistant.

Two tabs:
  - Chat     : conversation with MEMORY + honest "related lines" panel
  - Archivio : sortable table of all indexed invoices + CSV export
Sidebar: invoice counter + multi/ZIP upload + clear conversation.

Run from the PROJECT ROOT (not from src/):
    streamlit run src/app.py
"""

import io
import os
import zipfile
from pathlib import Path

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(page_title="Invoice RAG Assistant", page_icon="\U0001F9FE", layout="wide")

# Make the Anthropic key available whether we run locally (.env) or on
# Streamlit Cloud (Secrets). This MUST run before importing rag_chain, which
# creates the Anthropic client at import time.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass  # no secrets file locally -> rag_chain will read the .env instead

from parser import parse_invoice
from analytics import load_dataframes
from vectorstore import (
    build_index, build_chunk_text, build_metadata,
    DB_PATH, COLLECTION_NAME, EMBED_MODEL,
)
from rag_chain import answer

RAW_DIR = Path("data/raw_xml")


# --- heavy resources: load ONCE per session (model + index) ----------------
@st.cache_resource
def get_collection():
    """Open the existing ChromaDB collection, or build it if missing.
    Cached so the embedding model loads only once."""
    chroma = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    try:
        return chroma.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return build_index()


def add_invoices_to_index(collection, invoices):
    """Add already-parsed invoices to the live collection (idempotent upsert)."""
    docs, metas, ids = [], [], []
    for inv in invoices:
        for i, riga in enumerate(inv["righe"]):
            docs.append(build_chunk_text(inv, riga))
            metas.append(build_metadata(inv, riga))
            ids.append(f"{inv['file']}#{i}")
    if docs:
        collection.upsert(documents=docs, metadatas=metas, ids=ids)
    return len(invoices)


def archivio_view():
    """Return a display-friendly copy of the invoice table."""
    df = st.session_state.inv_df[
        ["numero", "data", "fornitore", "imponibile", "iva_totale",
         "totale", "regime", "pagamento", "ha_sconto"]
    ].rename(columns={
        "numero": "Numero", "data": "Data", "fornitore": "Fornitore",
        "imponibile": "Imponibile", "iva_totale": "IVA", "totale": "Totale",
        "regime": "Regime", "pagamento": "Pagamento", "ha_sconto": "Sconto",
    })
    return df.sort_values("Data")


collection = get_collection()

# --- session state ----------------------------------------------------------
if "inv_df" not in st.session_state:
    st.session_state.inv_df, st.session_state.line_df = load_dataframes()
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- sidebar: counter + upload ---------------------------------------------
with st.sidebar:
    st.header("Archivio fatture")
    st.metric("Fatture indicizzate", len(st.session_state.inv_df))

    st.divider()
    st.subheader("Aggiungi fatture")
    uploaded = st.file_uploader(
        "Carica file XML o uno ZIP di fatture",
        type=["xml", "zip"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Aggiungi all'archivio", type="primary"):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        new_paths = []
        for up in uploaded:
            name = up.name.lower()
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(up.getvalue())) as zf:
                    for member in zf.namelist():
                        if member.lower().endswith(".xml"):
                            target = RAW_DIR / Path(member).name
                            target.write_bytes(zf.read(member))
                            new_paths.append(target)
            elif name.endswith(".xml"):
                target = RAW_DIR / up.name
                target.write_bytes(up.getvalue())
                new_paths.append(target)

        new_invoices = []
        for p in new_paths:
            try:
                new_invoices.append(parse_invoice(p))
            except Exception as e:
                st.warning(f"Saltato {p.name}: {e}")

        if new_invoices:
            add_invoices_to_index(collection, new_invoices)
            st.session_state.inv_df, st.session_state.line_df = load_dataframes()
            st.success(f"Aggiunte {len(new_invoices)} fatture all'archivio.")
            st.rerun()
        else:
            st.info("Nessuna fattura XML valida trovata nei file caricati.")

    st.divider()
    if st.button("Cancella conversazione"):
        st.session_state.messages = []
        st.rerun()


# --- main area: two tabs ----------------------------------------------------
st.title("\U0001F9FE Invoice RAG Assistant")

tab_chat, tab_arch = st.tabs(["\U0001F4AC Chat", "\U0001F4CB Archivio"])

with tab_chat:
    st.caption("Interroga in linguaggio naturale il tuo archivio di fatture elettroniche.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Righe correlate (ricerca semantica)"):
                    for s in m["sources"]:
                        st.markdown(f"- {s}")

with tab_arch:
    df_view = archivio_view()
    st.subheader(f"Archivio fatture ({len(df_view)} indicizzate)")
    st.dataframe(df_view, use_container_width=True, hide_index=True)
    st.download_button(
        "\u2B07\uFE0F Scarica archivio (CSV)",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="archivio_fatture.csv",
        mime="text/csv",
    )


# --- chat input (root level -> stays pinned across tabs) --------------------
if prompt := st.chat_input("Es: quali fatture sono in reverse charge?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    with st.spinner("Sto cercando nelle fatture..."):
        testo, righe = answer(
            prompt, history,
            st.session_state.inv_df, st.session_state.line_df, collection,
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": testo, "sources": righe}
    )
    st.rerun()