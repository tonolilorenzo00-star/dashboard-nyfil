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

def load_evaluation(cliente: str, anno: str) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    question_keys = [q['key'] for q in EVALUATION_QUESTIONS]
    cursor.execute(f"SELECT {', '.join(question_keys)} FROM evaluation WHERE cliente = ? AND anno = ?", (cliente, anno))
    row = cursor.fetchone()
    if row: return dict(zip(question_keys, row))
    return {key: 3 for key in question_keys}

def save_evaluation(cliente: str, anno: str, data_dict: dict):
    try:
        conn = get_db_connection()
        question_keys = [q['key'] for q in EVALUATION_QUESTIONS]
        columns, placeholders = ", ".join(question_keys), ", ".join(["?"] * len(question_keys))
        query = f"INSERT OR REPLACE INTO evaluation (cliente, anno, {columns}, updated_at) VALUES (?, ?, {placeholders}, ?)"
        values = [cliente, anno] + [data_dict.get(key, 1) for key in question_keys] + [datetime.now().isoformat()]
        with conn:
            conn.execute(query, tuple(values))
        st.toast(f"Valutazione per {cliente.upper()} ({anno}) salvata!")
    except sqlite3.OperationalError as e:
        st.error(f"Errore di salvataggio: {e}. Sulla versione gratuita di Streamlit Cloud, il database potrebbe essere in sola lettura. Riprova più tardi.")

def calculate_scores(eval_data):
    total_score = sum(eval_data.values())
    val_economico_keys = [q['key'] for q in EVALUATION_QUESTIONS if q['category'] == 'Valore Economico']
    val_relazionale_keys = [q['key'] for q in EVALUATION_QUESTIONS if q['category'] == 'Valore Relazionale']
    val_economico = np.mean([eval_data[k] for k in val_economico_keys])
    val_relazionale = np.mean([eval_data[k] for k in val_relazionale_keys])
    return total_score, val_economico, val_relazionale

def get_matrix_quadrant(x, y):
    if x > 3 and y > 3: return "Partner Chiave"
    if x > 3 and y <= 3: return "Specialista Redditizio"
    if x <= 3 and y > 3: return "Amico a Basso Impatto"
    return "Cliente Marginale"

# --- PAGINE DELL'APPLICAZIONE ---

def page_dashboard(df_clienti, df_ordini, anni_selezionati, paese_selezionato, analysis_mode):
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
        if analysis_mode == "Aggrega Anni":
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
        else: # Confronta Anni
            st.info("Modalità Confronto Anni: andamento dei 10 articoli più importanti nel periodo.")
            top_10_articoli = ordini_filtrati_globale.groupby('ARTICOLO')['KG'].sum().nlargest(10).index
            df_compare = ordini_filtrati_globale[ordini_filtrati_globale['ARTICOLO'].isin(top_10_articoli)]
            pivot_table = df_compare.pivot_table(index='ARTICOLO', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
            pivot_table['Totale'] = pivot_table.sum(axis=1)
            pivot_table.sort_values('Totale', ascending=False, inplace=True)
            del pivot_table['Totale']
            st.dataframe(pivot_table.style.format("{:,.2f} Kg"), use_container_width=True)
    else:
        st.info("Seleziona almeno un anno per visualizzare i macrodati degli ordini.")


def page_elenco_clienti(df_clienti, df_ordini, anni_selezionati, paese_selezionato, analysis_mode):
    st.title("Elenco e Segmentazione Clienti")
    df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
    if paese_selezionato != "Tutti":
        df_filtrato = df_filtrato[df_filtrato['PAESE'] == paese_selezionato]
    
    st.subheader("Ranking Clienti")
    if not df_filtrato.empty:
        df_ranking_base = df_filtrato.groupby('CLIENTE').agg(PAESE=('PAESE', 'first')).reset_index()
        
        if analysis_mode == "Aggrega Anni":
            df_ranking = df_filtrato.groupby('CLIENTE').agg(Fatturato_Anagrafica=('FATTURATO', 'sum')).reset_index()
            if not df_ordini.empty and anni_selezionati:
                ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
                df_ordini_agg = ordini_filtrati.groupby('nome_cliente').agg(KG_Ordinati=('KG', 'sum')).reset_index()
                df_ranking = pd.merge(df_ranking, df_ordini_agg, left_on='CLIENTE', right_on='nome_cliente', how='left')
            df_ranking = pd.merge(df_ranking, df_ranking_base, on='CLIENTE', how='left')
            df_ranking['KG_Ordinati'] = df_ranking['KG_Ordinati'].fillna(0)
            df_ranking = df_ranking.sort_values('Fatturato_Anagrafica', ascending=False)
            df_ranking['CLIENTE_DISPLAY'] = df_ranking['CLIENTE'].str.upper()
            df_display = df_ranking[['CLIENTE_DISPLAY', 'PAESE', 'Fatturato_Anagrafica', 'KG_Ordinati']].copy()
            df_display.rename(columns={'CLIENTE_DISPLAY': 'CLIENTE'}, inplace=True)
            df_display['Fatturato_Anagrafica'] = df_display['Fatturato_Anagrafica'].apply(format_euro_robust)
            df_display['KG_Ordinati'] = df_display['KG_Ordinati'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#", "."))
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        else: # Modalità "Confronta Anni"
            st.info("Modalità Confronto Anni: le tabelle mostrano i dati disaggregati per anno.")
            fatturato_pivot = df_filtrato.pivot_table(index='CLIENTE', columns='ANNO', values='FATTURATO', aggfunc='sum')
            ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
            kg_pivot = ordini_filtrati.pivot_table(index='nome_cliente', columns='ANNO', values='KG', aggfunc='sum')
            df_compare = pd.merge(fatturato_pivot, kg_pivot, left_index=True, right_index=True, how='outer').fillna(0)
            
            col_order = []
            for year in sorted(anni_selezionati, reverse=True):
                col_fatt = f"Fatturato_{year}"
                col_kg = f"KG_{year}"
                df_compare.rename(columns={year+'_x': col_fatt, year+'_y': col_kg}, inplace=True, errors='ignore')
                if col_fatt in df_compare.columns:
                    df_compare[col_fatt] = df_compare[col_fatt].apply(format_euro_robust)
                if col_kg in df_compare.columns:
                    df_compare[col_kg] = df_compare[col_kg].apply(lambda x: f"{x:,.2f} Kg")
                col_order.extend([col for col in [col_fatt, col_kg] if col in df_compare.columns])

            df_compare = pd.merge(df_compare, df_ranking_base, left_index=True, right_on='CLIENTE', how='left').set_index('CLIENTE')
            df_compare.reset_index(inplace=True)
            df_compare['CLIENTE'] = df_compare['CLIENTE'].str.upper()
            
            st.dataframe(df_compare[['CLIENTE', 'PAESE'] + col_order], use_container_width=True, hide_index=True)

        with st.expander("Segmentazione Clienti (Basata sull'ultimo anno di valutazione)"):
            if not anni_selezionati:
                st.info("Seleziona un anno per visualizzare la segmentazione.")
            else:
                anno_segmentazione = anni_selezionati[0]
                all_evals = []
                clienti_unici_filtrati = df_filtrato['CLIENTE'].unique()
                for cliente in clienti_unici_filtrati:
                    eval_data = load_evaluation(cliente, anno_segmentazione)
                    if sum(eval_data.values()) != len(eval_data) * 3:
                        _, val_ec, val_rel = calculate_scores(eval_data)
                        segment = get_matrix_quadrant(val_ec, val_rel)
                        ha_valutazione = True
                    else:
                        segment = 'Valutazione non ancora avvenuta'
                        ha_valutazione = False
                    all_evals.append({'CLIENTE': cliente.upper(), 'VALUTAZIONE': segment, 'Ha_Valutazione': ha_valutazione})
                
                if all_evals:
                    df_segments = pd.DataFrame(all_evals)
                    df_segments.sort_values(by='Ha_Valutazione', ascending=False, inplace=True)
                    st.dataframe(df_segments[['CLIENTE', 'VALUTAZIONE']], use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Nessuna valutazione trovata per l'anno {anno_segmentazione}.")

def page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode):
    st.title("Analisi Dettagliata Cliente")
    
    clienti_options = sorted(df_clienti['CLIENTE'].str.upper().unique())
    clienti_selezionati_upper = st.multiselect(
        "Seleziona uno o più clienti per l'analisi", 
        options=clienti_options, 
        key='client_selector_detail'
    )
    
    if not clienti_selezionati_upper:
        st.info("Seleziona uno o più clienti per iniziare l'analisi.")
        return

    clienti_selezionati = [c.lower() for c in clienti_selezionati_upper]
    
    anni_scheda_selezionati = st.multiselect(
        "Seleziona anni per l'analisi di dettaglio", 
        options=anni_disponibili, 
        default=anni_selezionati_globali,
        key="anni_dettaglio_selector"
    )
    
    st.header(f"Scheda Alleati: {', '.join(clienti_selezionati_upper)}")
    
    anno_riferimento_scheda = anni_scheda_selezionati[0] if anni_scheda_selezionati else anni_disponibili[0]
    
    tab_eval, tab_dati, tab_ordini = st.tabs(["Valutazione Alleati", "Anagrafica & Fatturato", "Ordini & Statistiche"])
    
    with tab_eval:
        st.subheader(f"Valutazioni Individuali (Anno di riferimento: {anno_riferimento_scheda})")
        evals_data = {}
        for cliente in clienti_selezionati:
            with st.expander(f"Valutazione per {cliente.upper()}"):
                with st.form(key=f"evaluation_form_{cliente}_{anno_riferimento_scheda}"):
                    eval_data = load_evaluation(cliente, anno_riferimento_scheda)
                    cols = st.columns(3)
                    temp_eval_data = {}
                    for i, q in enumerate(EVALUATION_QUESTIONS):
                        with cols[i % 3]:
                            temp_eval_data[q['key']] = st.slider(
                                q['text'], 1, 5, value=eval_data.get(q['key'], 3), key=f"{q['key']}_{cliente}_{anno_riferimento_scheda}"
                            )
                    submitted = st.form_submit_button("Salva Valutazione")
                    if submitted:
                        save_evaluation(cliente, anno_riferimento_scheda, temp_eval_data)
                        st.cache_data.clear()
                        st.rerun()
                evals_data[cliente] = load_evaluation(cliente, anno_riferimento_scheda)

        st.divider()
        st.subheader("Analisi Strategica Comparata")
        fig_matrix = go.Figure()
        fig_radar = go.Figure()
        for cliente, data in evals_data.items():
            total_score, val_economico, val_relazionale = calculate_scores(data)
            fig_matrix.add_trace(go.Scatter(x=[val_economico], y=[val_relazionale], mode='markers+text', text=cliente.upper(), marker=dict(size=15), name=cliente.upper()))
            radar_values = [data[q['key']] for q in EVALUATION_QUESTIONS]
            fig_radar.add_trace(go.Scatterpolar(r=radar_values + [radar_values[0]], theta=[f"Q{i+1}" for i in range(15)] + ["Q1"], fill='toself', name=cliente.upper(), opacity=0.7))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Matrice Decisionale")
            st.plotly_chart(fig_matrix, use_container_width=True)
        with col2:
            st.markdown("##### Profili Radar")
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab_dati:
        for cliente in clienti_selezionati:
            with st.expander(f"Dati per {cliente.upper()}"):
                dati_cliente = df_clienti[df_clienti['CLIENTE'] == cliente]
                st.subheader(f"Anagrafica (Riferimento anno: {anno_riferimento_scheda})")
                
                anagrafica_anno_scheda = dati_cliente[dati_cliente['ANNO'] == anno_riferimento_scheda]
                if not anagrafica_anno_scheda.empty:
                    anagrafica = anagrafica_anno_scheda.iloc[0]
                elif not dati_cliente.empty:
                    anagrafica = dati_cliente.sort_values('ANNO', ascending=False).iloc[0]
                    st.info(f"Dati anagrafici per l'anno {anno_riferimento_scheda} non trovati. Mostro i più recenti.")
                else:
                    st.warning("Dati anagrafici non disponibili.")
                    continue

                cols_anagrafica = st.columns(3)
                cols_anagrafica[0].markdown(f"**Indirizzo:**<br>{anagrafica.get('VIA', 'N/D')}", unsafe_allow_html=True)
                cols_anagrafica[1].markdown(f"**Paese:**<br>{anagrafica.get('PAESE', 'N/D')}", unsafe_allow_html=True)
                cols_anagrafica[2].markdown(f"**Contatti:**<br>Titolare: {anagrafica.get('TITOLARE', 'N/D')}", unsafe_allow_html=True)
                st.divider()
                st.subheader("Andamento Fatturato Annuale (da Anagrafica)")
                fatturato_annuale = dati_cliente.groupby('ANNO')['FATTURATO'].sum().sort_index()
                fig_bar = go.Figure(data=[go.Bar(x=fatturato_annuale.index, y=fatturato_annuale.values, text=[format_euro_robust(v) for v in fatturato_annuale.values], textposition='auto')])
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab_ordini:
        st.subheader(f"Statistiche Ordini (Anni selezionati: {', '.join(anni_scheda_selezionati)})")
        if df_ordini.empty or not anni_scheda_selezionati:
            st.info("Seleziona uno o più anni nel selettore qui sopra.")
        else:
            ordini_selezionati = df_ordini[(df_ordini['ANNO'].isin(anni_scheda_selezionati)) & (df_ordini['nome_cliente'].isin(clienti_selezionati))]
            if ordini_selezionati.empty:
                st.info("Nessun ordine trovato per i clienti e gli anni selezionati.")
            
            elif analysis_mode == "Aggrega Anni":
                st.subheader("Statistiche Aggregate (Clienti Selezionati)")
                total_kg, total_fatturato_ordini = ordini_selezionati['KG'].sum(), ordini_selezionati['FATTURATO_ORDINE'].sum()
                prezzo_medio_kg = (total_fatturato_ordini / total_kg) if total_kg > 0 else 0
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Kg Totali (Aggregati)", f"{total_kg:,.2f} Kg".replace(",", "."))
                kpi2.metric("Fatturato Ordini (Aggregato)", format_euro_robust(total_fatturato_ordini))
                kpi3.metric("Prezzo Medio Kg (Aggregato)", f"{format_euro_robust(prezzo_medio_kg)} /Kg")
                kpi4.metric("N. Righe Ordine (Aggregate)", f"{len(ordini_selezionati)}")
                st.divider()

                st.subheader("Dettaglio per Cliente")
                for cliente in clienti_selezionati:
                    with st.expander(f"Ordini per {cliente.upper()}"):
                        ordini_cliente_singolo = ordini_selezionati[ordini_selezionati['nome_cliente'] == cliente]
                        if ordini_cliente_singolo.empty:
                            st.write("Nessun dato per questo cliente nel periodo selezionato.")
                            continue
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### Top 5 Articoli per Kg")
                            agg_articolo_chart = ordini_cliente_singolo.groupby('ARTICOLO').agg(Totale_Kg=('KG', 'sum')).nlargest(5, 'Totale_Kg').reset_index()
                            fig_pie_art = go.Figure(data=[go.Pie(labels=agg_articolo_chart['ARTICOLO'], values=agg_articolo_chart['Totale_Kg'], hole=.3, textinfo='percent+label')])
                            st.plotly_chart(fig_pie_art, use_container_width=True)
                        with col2:
                            st.markdown("##### Top 5 Colori per Kg")
                            agg_colore_chart = ordini_cliente_singolo.groupby('COLORE').agg(Totale_Kg=('KG', 'sum')).nlargest(5, 'Totale_Kg').reset_index()
                            fig_pie_col = go.Figure(data=[go.Pie(labels=agg_colore_chart['COLORE'], values=agg_colore_chart['Totale_Kg'], hole=.3, textinfo='percent+label')])
                            st.plotly_chart(fig_pie_col, use_container_width=True)
                        
                        st.divider()

                        def display_agg_table(df_agg, title, filename_prefix, key_suffix):
                            st.markdown(f"##### {title}")
                            column_config = {"Totale_Kg": st.column_config.NumberColumn("Totale Kg", format="%.2f Kg")}
                            if 'Totale_Fatturato' in df_agg.columns:
                                column_config["Totale_Fatturato"] = st.column_config.NumberColumn("Totale Fatturato", format="€ %.2f")
                            st.dataframe(df_agg, use_container_width=True, hide_index=True, column_config=column_config)
                            csv = df_agg.to_csv(index=False, sep=';', decimal=',', encoding='latin1')
                            st.download_button(f"📥 Export {title}", csv, f"{filename_prefix}_{cliente}.csv", "text/csv", key=f"btn_{filename_prefix}_{key_suffix}_{'_'.join(anni_scheda_selezionati)}")

                        agg_articolo_full = ordini_cliente_singolo.groupby('ARTICOLO').agg(Totale_Kg=('KG', 'sum'), Totale_Fatturato=('FATTURATO_ORDINE', 'sum')).reset_index().sort_values('Totale_Kg', ascending=False)
                        display_agg_table(agg_articolo_full, "Dettaglio Analisi per Articolo", "analisi_articolo", cliente)

                        agg_colore_full = ordini_cliente_singolo.groupby('COLORE').agg(Totale_Kg=('KG', 'sum'), Totale_Fatturato=('FATTURATO_ORDINE', 'sum')).reset_index().sort_values('Totale_Kg', ascending=False)
                        display_agg_table(agg_colore_full, "Dettaglio Analisi per Colore", "analisi_colore", cliente)

                        agg_articolo_colore_full = ordini_cliente_singolo.groupby(['ARTICOLO', 'COLORE']).agg(Totale_Kg=('KG', 'sum'), Totale_Fatturato=('FATTURATO_ORDINE', 'sum')).reset_index().sort_values('Totale_Kg', ascending=False)
                        display_agg_table(agg_articolo_colore_full, "Dettaglio Analisi per Articolo e Colore", "analisi_articolo_colore", cliente)

            else: # Confronta Anni
                st.info("Modalità Confronto Anni: le tabelle mostrano i dati disaggregati per anno.")
                for cliente in clienti_selezionati:
                    with st.expander(f"Confronto ordini per {cliente.upper()}"):
                        ordini_cliente_singolo = ordini_selezionati[ordini_selezionati['nome_cliente'] == cliente]
                        if ordini_cliente_singolo.empty:
                            st.write("Nessun dato per questo cliente nel periodo.")
                            continue

                        st.markdown("##### Confronto Annuale per Articolo")
                        pivot_kg_art = ordini_cliente_singolo.pivot_table(index='ARTICOLO', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
                        pivot_kg_art['Totale'] = pivot_kg_art.sum(axis=1)
                        st.dataframe(pivot_kg_art.sort_values('Totale', ascending=False).drop(columns='Totale').style.format("{:,.2f}"), use_container_width=True)

                        st.markdown("##### Confronto Annuale per Colore")
                        pivot_kg_col = ordini_cliente_singolo.pivot_table(index='COLORE', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
                        pivot_kg_col['Totale'] = pivot_kg_col.sum(axis=1)
                        st.dataframe(pivot_kg_col.sort_values('Totale', ascending=False).drop(columns='Totale').style.format("{:,.2f}"), use_container_width=True)


def page_stato_dati(df_clienti, df_ordini):
    st.title("Stato dei Dati e Diagnostica")
    st.header("1. Controllo File")
    st.write("Questi sono i file che l'applicazione ha trovato nella cartella `data/`:")
    files_trovati = [p.name for p in DATA_DIR.glob('*')]
    if files_trovati:
        st.dataframe(files_trovati, use_container_width=True)
    else:
        st.error("Nessun file trovato nella cartella 'data'. Assicurati di aver caricato i file su GitHub.")

    st.header("2. Analisi del Caricamento")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anagrafica Clienti (`elenco clienti.csv`)")
        if not df_clienti.empty:
            st.metric("Righe totali caricate (dopo espansione anni)", len(df_clienti))
            st.metric("Clienti unici trovati", df_clienti['CLIENTE'].nunique())
        else:
            st.warning("Il file dell'anagrafica clienti non è stato caricato o è vuoto.")

    with col2:
        st.subheader("File Ordini (`ordini_*.csv`)")
        if not df_ordini.empty:
            st.metric("Righe totali caricate", len(df_ordini))
            st.metric("Clienti unici trovati", df_ordini['nome_cliente'].nunique())
        else:
            st.warning("Nessun file degli ordini caricato o sono tutti vuoti.")
            
    st.header("3. Diagnosi delle Corrispondenze")
    if not df_clienti.empty and not df_ordini.empty:
        clienti_anagrafica = set(df_clienti['CLIENTE'].unique())
        clienti_ordini = set(df_ordini['nome_cliente'].unique())
        
        clienti_corrispondenti = clienti_anagrafica.intersection(clienti_ordini)
        clienti_orfani = clienti_ordini - clienti_anagrafica

        st.metric("Numero di clienti che corrispondono tra Anagrafica e Ordini", len(clienti_corrispondenti))
        
        if clienti_orfani:
            st.error(f"Trovati {len(clienti_orfani)} clienti 'orfani'!")
            st.write("Questi clienti sono presenti nei file degli ordini, ma **NON** nel file `elenco clienti.csv` (o i nomi non corrispondono esattamente). Questo è il motivo per cui il loro fatturato non viene visualizzato.")
            st.write("**Azione richiesta:** Correggi i nomi di questi clienti nel file `elenco clienti.csv` per farli corrispondere esattamente a come appaiono qui sotto, poi ricarica il file su GitHub.")
            st.dataframe(sorted([c.upper() for c in clienti_orfani]), use_container_width=True)
        else:
            st.success("Ottimo! Tutti i clienti presenti negli ordini hanno una corrispondenza nel file anagrafico.")
    else:
        st.info("Carica sia il file anagrafica che i file ordini per eseguire la diagnosi.")

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
    page_dashboard(df_clienti, df_ordini, anni_selezionati_globali, paese_selezionato, analysis_mode)
elif pagina_selezionata == "Elenco Clienti":
    page_elenco_clienti(df_clienti, df_ordini, anni_selezionati_globali, paese_selezionato, analysis_mode)
elif pagina_selezionata == "Analisi Dettagliata":
    page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode)
elif pagina_selezionata == "Stato dei Dati":
    page_stato_dati(df_clienti, df_ordini)
