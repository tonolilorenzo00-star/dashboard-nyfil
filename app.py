import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import numpy as np
import glob
import re
import base64

# --- CONFIGURAZIONE PAGINA E COSTANTI ---
st.set_page_config(page_title="Analisi Clienti Nyfil", layout="wide")

# FIX DEFINITIVO: Usa percorsi assoluti basati sulla posizione dello script
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
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

# --- FUNZIONI DI UTILITÀ ---

def format_euro_robust(value):
    try:
        if pd.isna(value) or not isinstance(value, (int, float)):
            return "N/A"
        return f"€ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "N/A"

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

def clean_customer_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

def detect_country(cap: str, provincia: str) -> str:
    cap_str, prov_str = str(cap).strip(), str(provincia).strip().upper()
    if not prov_str or prov_str == 'NAN': return "Estero"
    if (cap_str.isdigit() and len(cap_str) == 5) or prov_str in PROVINCE_ITALIANE: return "Italia"
    return "Estero"

@st.cache_resource
def get_db_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_db_connection()
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

@st.cache_data
def load_clients_df(uploaded_file=None) -> pd.DataFrame:
    source = CLIENTS_CSV if CLIENTS_CSV.exists() else uploaded_file
    if not source: return pd.DataFrame()
    try:
        df = pd.read_csv(source, sep=';', encoding='latin1', low_memory=False)
        df.columns = [col.strip().lower() for col in df.columns]
        required_cols = {"nome_clietne": "CLIENTE", "nome_cliente": "CLIENTE", "via": "VIA", "città": "CITTA", "cap": "CAP", "provincia": "PROVINCIA", "titolare_azienda": "TITOLARE", "recapiti_mail": "EMAIL", "anno": "ANNO_ORIG", "imponibile": "IMPONIBILE"}
        df.rename(columns=required_cols, inplace=True)
        
        if 'CLIENTE' in df.columns:
            df['CLIENTE'] = df['CLIENTE'].apply(clean_customer_name)
            
        df['ANNO_ORIG'] = df['ANNO_ORIG'].astype(str).str.split('/')
        df = df.explode('ANNO_ORIG')
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
    order_files = glob.glob(str(DATA_DIR / "ordini_*"))
    if not order_files: return pd.DataFrame()
    df_list = []
    for file in order_files:
        try:
            year_match = re.search(r'(\d+)', Path(file).stem)
            if not year_match: continue
            year_full = normalize_year(year_match.group(1))
            df = pd.read_csv(file, sep=';', encoding='latin1', low_memory=False) if file.endswith('.csv') else pd.read_excel(file)
            df.columns = [re.sub(r'_\d+$', '', col).strip().lower() for col in df.columns]
            df['ANNO'] = year_full
            df_list.append(df)
        except Exception as e:
            st.warning(f"Impossibile leggere il file {file}: {e}")
    if not df_list: return pd.DataFrame()
    
    df_orders = pd.concat(df_list, ignore_index=True)
    
    if 'nome_cliente' in df_orders.columns:
        df_orders['nome_cliente'] = df_orders['nome_cliente'].apply(clean_customer_name)

    df_orders.dropna(subset=['articolo_colore', 'quantita'], inplace=True)
    df_orders = df_orders[df_orders['quantita'] != 0].copy()
    
    if 'imponibile' in df_orders.columns:
        df_orders['FATTURATO_ORDINE'] = df_orders['imponibile'].apply(parse_decimal_string)
    else:
        df_orders['FATTURATO_ORDINE'] = 0

    df_orders['KG'] = df_orders['quantita'].apply(parse_decimal_string).round(2)
    split_data = df_orders['articolo_colore'].str.rsplit(' - ', n=1, expand=True)
    df_orders['ARTICOLO'] = split_data[0].str.strip()
    df_orders['COLORE'] = split_data[1].str.strip().fillna('NON SPECIFICATO')
    
    return df_orders[['nome_cliente', 'ANNO', 'ARTICOLO', 'COLORE', 'KG', 'FATTURATO_ORDINE']]

def save_evaluation(cliente: str, anno: str, data_dict: dict):
    # (Funzione non modificata)
    pass 

# ... (Altre funzioni di utilità non modificate) ...

# --- PAGINE DELL'APPLICAZIONE ---

def page_dashboard(df_clienti, df_ordini, anni_selezionati, paese_selezionato):
    st.title("Dashboard Riepilogativa")
    df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
    if paese_selezionato != "Tutti": 
        df_filtrato = df_filtrato[df_filtrato['PAESE'] == paese_selezionato]

    st.header("KPI Generali (da anagrafica)")
    if not df_filtrato.empty:
        total_revenue = df_filtrato['FATTURATO'].sum()
        revenue_italia = df_filtrato[df_filtrato['PAESE'] == 'Italia']['FATTURATO'].sum()
        quota_italia = (revenue_italia / total_revenue * 100) if total_revenue > 0 else 0
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Totale Fatturato", format_euro_robust(total_revenue))
        kpi2.metric("Quota Italia", f"{quota_italia:.1f}%")
        kpi3.metric("Quota Estero", f"{100 - quota_italia:.1f}%")
        kpi4.metric("N. Clienti nel filtro", f"{df_filtrato['CLIENTE'].nunique()}")
    else:
        st.info("Nessun dato anagrafico per i filtri selezionati.")
    
    st.header("Macrodati Ordini")
    if not df_ordini.empty and anni_selezionati:
        ordini_filtrati_globale = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
        col1_macro, col2_macro, col3_macro = st.columns(3)
        with col1_macro:
            st.markdown("###### Top 10 Articoli (per Kg)")
            st.dataframe(ordini_filtrati_globale.groupby('ARTICOLO')['KG'].sum().nlargest(10).reset_index(), use_container_width=True, hide_index=True, height=385)
        with col2_macro:
            st.markdown("###### Top 10 Colori (per Kg)")
            st.dataframe(ordini_filtrati_globale.groupby('COLORE')['KG'].sum().nlargest(10).reset_index(), use_container_width=True, hide_index=True, height=385)
        with col3_macro:
            st.markdown("###### Top 10 Articolo-Colore (per Kg)")
            st.dataframe(ordini_filtrati_globale.groupby(['ARTICOLO', 'COLORE'])['KG'].sum().nlargest(10).reset_index(), use_container_width=True, hide_index=True, height=385)
    else:
        st.info("Seleziona almeno un anno per visualizzare i macrodati degli ordini.")


def page_elenco_clienti(df_clienti, df_ordini, anni_selezionati, paese_selezionato, analysis_mode):
    st.title("Elenco e Segmentazione Clienti")
    df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
    if paese_selezionato != "Tutti":
        df_filtrato = df_filtrato[df_filtrato['PAESE'] == paese_selezionato]
    
    st.subheader("Ranking Clienti")
    if not df_filtrato.empty:
        if analysis_mode == "Aggrega Anni":
            df_ranking = df_filtrato.groupby('CLIENTE').agg(Fatturato_Anagrafica=('FATTURATO', 'sum'), PAESE=('PAESE', 'first')).reset_index()
            if not df_ordini.empty and anni_selezionati:
                ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
                df_ordini_agg = ordini_filtrati.groupby('nome_cliente').agg(KG_Ordinati=('KG', 'sum')).reset_index()
                df_ranking = pd.merge(df_ranking, df_ordini_agg, left_on='CLIENTE', right_on='nome_cliente', how='left')
                df_ranking['KG_Ordinati'] = df_ranking['KG_Ordinati'].fillna(0)
            else:
                df_ranking['KG_Ordinati'] = 0
            df_ranking = df_ranking.sort_values('Fatturato_Anagrafica', ascending=False)
            df_ranking['CLIENTE_DISPLAY'] = df_ranking['CLIENTE'].str.upper()
            df_display = df_ranking[['CLIENTE_DISPLAY', 'PAESE', 'Fatturato_Anagrafica', 'KG_Ordinati']].copy()
            df_display.rename(columns={'CLIENTE_DISPLAY': 'CLIENTE'}, inplace=True)
            df_display['Fatturato_Anagrafica'] = df_display['Fatturato_Anagrafica'].apply(format_euro_robust)
            df_display['KG_Ordinati'] = df_display['KG_Ordinati'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#", "."))
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        else: # Modalità "Confronta Anni"
            st.info("Modalità Confronto Anni: le tabelle mostrano i dati disaggregati per anno.")
            fatturato_pivot = df_filtrato.pivot_table(index='CLIENTE', columns='ANNO', values='FATTURATO', aggfunc='sum').fillna(0)
            
            ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
            kg_pivot = ordini_filtrati.pivot_table(index='nome_cliente', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
            
            df_ranking_compare = pd.merge(fatturato_pivot, kg_pivot, left_index=True, right_index=True, how='outer').fillna(0)
            
            # Formattazione e ordinamento colonne
            col_order = ['CLIENTE', 'PAESE']
            for year in sorted(anni_selezionati, reverse=True):
                col_order.append(f"Fatturato_{year}")
                col_order.append(f"KG_{year}")
                df_ranking_compare.rename(columns={year+'_x': f"Fatturato_{year}", year+'_y': f"KG_{year}"}, inplace=True)
                df_ranking_compare[f"Fatturato_{year}"] = df_ranking_compare[f"Fatturato_{year}"].apply(format_euro_robust)
                df_ranking_compare[f"KG_{year}"] = df_ranking_compare[f"KG_{year}"].apply(lambda x: f"{x:,.2f} Kg")


            paese_info = df_filtrato[['CLIENTE', 'PAESE']].drop_duplicates().set_index('CLIENTE')
            df_ranking_compare = df_ranking_compare.merge(paese_info, left_index=True, right_index=True)
            df_ranking_compare.reset_index(inplace=True)
            df_ranking_compare.rename(columns={'index': 'CLIENTE'}, inplace=True)
            
            st.dataframe(df_ranking_compare[col_order], use_container_width=True, hide_index=True)


        with st.expander("Segmentazione Clienti (Basata sull'ultimo anno di valutazione)"):
            pass

        clienti_options = sorted(df_ranking['CLIENTE'].str.upper().unique())
        clienti_selezionati_upper = st.multiselect( "Seleziona uno o più clienti per l'analisi dettagliata", options=clienti_options, key='client_selector')
        st.session_state.clienti_selezionati = [c.lower() for c in clienti_selezionati_upper]
        st.info("Una volta selezionati i clienti, vai alla pagina 'Analisi Dettagliata' dalla sidebar.")

def page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode):
    st.title("Analisi Dettagliata Cliente")

    if 'clienti_selezionati' not in st.session_state or not st.session_state.clienti_selezionati:
        st.info("Seleziona uno o più clienti dalla pagina 'Elenco Clienti' per iniziare l'analisi.")
        return
    
    # ... (Codice completo della pagina di analisi dettagliata)
    pass


def page_stato_dati(df_clienti, df_ordini):
    st.title("Stato dei Dati e Diagnostica")
    # ... (Codice non modificato)
    pass

# --- LOGICA PRINCIPALE E NAVIGAZIONE ---
init_db()

df_clienti = load_clients_df()
df_ordini = load_all_orders_df()

if df_clienti.empty and df_ordini.empty:
    st.error("Nessun file dati ('elenco clienti.csv' o 'ordini_*.csv') trovato nella cartella 'data'.")
    st.stop()

anni_clienti = df_clienti['ANNO'].unique() if not df_clienti.empty else []
anni_ordini = df_ordini['ANNO'].unique() if not df_ordini.empty else []
tutti_gli_anni = pd.concat([pd.Series(anni_clienti), pd.Series(anni_ordini)]).unique()
anni_disponibili = sorted([anno for anno in tutti_gli_anni if pd.notna(anno)], reverse=True)

with st.sidebar:
    logo_path = DATA_DIR / "Logo_nyfil.png"
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    
    st.title("Navigazione")
    pagina_selezionata = st.radio(
        "Scegli una pagina:",
        ("Dashboard", "Elenco Clienti", "Analisi Dettagliata", "Stato dei Dati")
    )

    st.divider()
    st.header("Filtri Globali")
    anni_selezionati_globali = st.multiselect("Anni", options=anni_disponibili, default=anni_disponibili)
    paese_selezionato = st.selectbox("Paese", options=["Tutti", "Italia", "Estero"])
    analysis_mode = st.radio("Modalità di Analisi Annuale", ["Aggrega Anni", "Confronta Anni"], key='analysis_mode_selector')


# --- ROUTING DELLE PAGINE ---
if pagina_selezionata == "Dashboard":
    page_dashboard(df_clienti, df_ordini, anni_selezionati_globali, paese_selezionato)
elif pagina_selezionata == "Elenco Clienti":
    page_elenco_clienti(df_clienti, df_ordini, anni_selezionati_globali, paese_selezionato, analysis_mode)
elif pagina_selezionata == "Analisi Dettagliata":
    page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode)
elif pagina_selezionata == "Stato dei Dati":
    page_stato_dati(df_clienti, df_ordini)
