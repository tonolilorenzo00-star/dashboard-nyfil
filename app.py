import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import locale
import numpy as np
import glob
import re
import base64

# --- CONFIGURAZIONE PAGINA E COSTANTI ---
st.set_page_config(page_title="Analisi Clienti Nyfil", layout="wide")

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "app.db"
CLIENTS_CSV = DATA_DIR / "elenco clienti.csv"

DATA_DIR.mkdir(exist_ok=True)

PROVINCE_ITALIANE = [
    "AG", "AL", "AN", "AO", "AR", "AP", "AT", "AV", "BA", "BT", "BL", "BN", "BG", "BI", "BO", "BZ", "BS", "CA", "CL", "CB", "CE", "CH", "CO", "CS", "CR", "KR", "CN", "EN", "FE", "FI", "FG", "FC", "FR", "GE", "GO", "GR", "IM", "IS", "SP", "AQ", "LT", "LE", "LC", "LI", "LO", "LU", "MC", "MN", "MS", "MT", "ME", "MI", "MO", "MB", "NA", "NO", "NU", "OR", "PD", "PA", "PR", "PV", "PG", "PU", "PE", "PC", "PI", "PT", "PN", "PZ", "PO", "RG", "RA", "RC", "RE", "RI", "RN", "RM", "RO", "SA", "SS", "SV", "SI", "SO", "SR", "TA", "TE", "TR", "TO", "TP", "TN", "TV", "TS", "UD", "VA", "VE", "VB", "VC", "VR", "VV", "VI", "VT"
]

EVALUATION_QUESTIONS = [
    {"key": "q1", "text": "1. GENERA FATTURATI IMPORTANTI PER NOI?", "category": "Valore Economico"},
    {"key": "q2", "text": "2. DIMOSTRA INTERESSE E COMPRENDE L’UTILITÀ DEL NS. PRODOTTO/SERVIZIO?", "category": "Valore Relazionale"},
    {"key": "q3", "text": "3. SIAMO IMPORTANTI PER LUI, CI VALUTA POSITIVAMENTE, TIENE A NOI?", "category": "Valore Relazionale"},
    {"key": "q4", "text": "4. RICHIEDE SFORZI E COSTI LOGISTICI?", "category": "Costi e Sforzi"},
    {"key": "q5", "text": "5. SI E' CREATO UN BUON LIVELLO DI EMPATIA?", "category": "Valore Relazionale"},
    {"key": "q6", "text": "6. E' SODDISFATTO DEL VALORE CHE RIUSCIAMO AD ASSICURARGLI?", "category": "Valore Relazionale"},
    {"key": "q7", "text": "7. AGGIUNGE PRESTIGIO AL NOSTRO PARCO ALLEATI?", "category": "Potenziale Futuro"},
    {"key": "q8", "text": "8. ABBIAMO CON LUI UN RAPPORTO DURATURO NEL TEMPO?", "category": "Valore Relazionale"},
    {"key": "q9", "text": "9. HA UNA BUONA FREQUENZA DI LAVORI?", "category": "Valore Economico"},
    {"key": "q10", "text": "10. RIUSCIAMO A GENERARE SODDISFAZIONE?", "category": "Valore Relazionale"},
    {"key": "q11", "text": "11. CI AIUTA AD ACQUISIRE NUOVI CLIENTI TRAMITE PASSAPAROLA?", "category": "Potenziale Futuro"},
    {"key": "q12", "text": "12. LE SUE CARATTERISTICHE LO RENDONO UN ALLEATO SVILUPPABILE?", "category": "Potenziale Futuro"},
    {"key": "q13", "text": "13. HA CLIENTI CON BUON POTENZIALE ECONOMICO?", "category": "Valore Economico"},
    {"key": "q14", "text": "14. ABBIAMO UNA BUONA LEADERSHIP NEI SUOI CONFRONTI?", "category": "Valore Relazionale"},
    {"key": "q15", "text": "15. E' UN PARTNER CON IL QUALE POTER CONDIVIDERE LA NOSTRA STRATEGIA?", "category": "Potenziale Futuro"},
]

try:
    locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')
except locale.Error:
    st.warning("Localizzazione italiana non trovata.")

# --- FUNZIONI DI UTILITÀ ---

def format_euro(value):
    try: return locale.currency(value, symbol='€', grouping=True)
    except (TypeError, ValueError): return "N/A"

def parse_decimal_string(s: str) -> float:
    if not isinstance(s, str): s = str(s)
    try: return float(s.replace('.', '').replace(',', '.'))
    except (ValueError, TypeError): return 0.0

def normalize_year(s: str) -> str:
    s = str(s).strip()
    if s.isdigit():
        if len(s) == 2: return f"20{s}"
        if len(s) == 4: return s
    return s

def detect_country(cap: str, provincia: str) -> str:
    cap_str, prov_str = str(cap).strip(), str(provincia).strip().upper()
    if not prov_str or prov_str == 'NAN': return "Estero"
    if (cap_str.isdigit() and len(cap_str) == 5) or prov_str in PROVINCE_ITALIANE: return "Italia"
    return "Estero"

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation (
                cliente TEXT NOT NULL, anno TEXT NOT NULL,
                q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER, q5 INTEGER,
                q6 INTEGER, q7 INTEGER, q8 INTEGER, q9 INTEGER, q10 INTEGER,
                q11 INTEGER, q12 INTEGER, q13 INTEGER, q14 INTEGER, q15 INTEGER,
                updated_at TEXT, PRIMARY KEY (cliente, anno)
            );
        """)
    return conn

@st.cache_data
def load_clients_df(uploaded_file=None) -> pd.DataFrame:
    source = CLIENTS_CSV if CLIENTS_CSV.exists() else uploaded_file
    if not source: return pd.DataFrame()
    try:
        df = pd.read_csv(source, sep=';', encoding='latin1')
        df.columns = [col.strip().lower() for col in df.columns]
        required_cols = {"nome_cliente": "CLIENTE", "via": "VIA", "città": "CITTA", "cap": "CAP", "provincia": "PROVINCIA", "titolare_azienda": "TITOLARE", "recapiti_mail": "EMAIL", "anno": "ANNO_ORIG", "imponibile": "IMPONIBILE"}
        df.rename(columns=required_cols, inplace=True)
        # FIX DEFINITIVO: Standardizza i nomi dei clienti (lowercase, no spazi)
        if 'CLIENTE' in df.columns:
            df['CLIENTE'] = df['CLIENTE'].str.strip().str.lower()
            
        df['ANNO'] = df['ANNO_ORIG'].apply(normalize_year)
        df['FATTURATO'] = df['IMPONIBILE'].apply(parse_decimal_string)
        df['PAESE'] = df.apply(lambda row: detect_country(row.get('CAP', ''), row.get('PROVINCIA', '')), axis=1)
        agg_cols = {k: 'first' for k in ['VIA', 'CITTA', 'CAP', 'PROVINCIA', 'TITOLARE', 'EMAIL', 'PAESE']}
        agg_cols['FATTURATO'] = 'sum'
        cols_to_agg = {k: v for k, v in agg_cols.items() if k in df.columns}
        return df.groupby(['CLIENTE', 'ANNO']).agg(cols_to_agg).reset_index()
    except Exception as e:
        st.error(f"Errore lettura file clienti: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all_orders_df() -> pd.DataFrame:
    order
