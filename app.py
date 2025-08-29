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
            df = pd.read_csv(file, sep=';', encoding='latin1') if file.endswith('.csv') else pd.read_excel(file)
            df.columns = [re.sub(r'_\d+$', '', col).strip().lower() for col in df.columns]
            df['ANNO'] = year_full
            df_list.append(df)
        except Exception as e:
            st.warning(f"Impossibile leggere il file {file}: {e}")
    if not df_list: return pd.DataFrame()
    
    df_orders = pd.concat(df_list, ignore_index=True)
    
    df_orders.dropna(subset=['articolo_colore', 'quantita'], inplace=True)
    df_orders = df_orders[df_orders['quantita'] != 0].copy()
    
    # Assicurarsi che la colonna 'imponibile' esista prima di usarla
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
    conn = get_db_connection()
    question_keys = [q['key'] for q in EVALUATION_QUESTIONS]
    columns, placeholders = ", ".join(question_keys), ", ".join(["?"] * len(question_keys))
    query = f"INSERT OR REPLACE INTO evaluation (cliente, anno, {columns}, updated_at) VALUES (?, ?, {placeholders}, ?)"
    values = [cliente, anno] + [data_dict.get(key, 1) for key in question_keys] + [datetime.now().isoformat()]
    with conn:
        conn.execute(query, tuple(values))
    st.toast(f"Valutazione per {cliente} ({anno}) salvata!")

# --- FUNZIONI PER NUOVA ANALISI ---

def calculate_scores(eval_data):
    total_score = sum(eval_data.values())
    val_economico_keys = [q['key'] for q in EVALUATION_QUESTIONS if q['category'] == 'Valore Economico']
    val_relazionale_keys = [q['key'] for q in EVALUATION_QUESTIONS if q['category'] == 'Valore Relazionale']
    val_economico = np.mean([eval_data[k] for k in val_economico_keys])
    val_relazionale = np.mean([eval_data[k] for k in val_relazionale_keys])
    return total_score, val_economico, val_relazionale

def get_customer_segment(score):
    if score >= 60: return "Cliente Oro", "🥇"
    if 40 <= score < 60: return "Cliente da Sviluppare", "📈"
    return "Cliente da Rivalutare", "⚠️"

def get_matrix_quadrant(x, y):
    if x > 3 and y > 3: return "Partner Chiave"
    if x > 3 and y <= 3: return "Specialista Redditizio"
    if x <= 3 and y > 3: return "Amico a Basso Impatto"
    return "Cliente Marginale"

def get_actionable_insights(eval_data, val_ec, val_rel):
    insights = []
    if eval_data['q1'] >= 4 and val_rel <= 2.5:
        insights.append(("Cliente ad alto fatturato ma bassa relazione", "Azioni di customer care dedicato, proporre servizi extra, programma loyalty, monitoraggio trimestrale."))
    if eval_data['q1'] <= 2 and val_rel >= 4:
        insights.append(("Cliente a basso fatturato ma alta relazione (Ambasciatore Potenziale)", "Offrire pacchetti entry-level, usarlo come testimonial, coinvolgerlo in eventi, valutare prodotti pilota."))
    if (eval_data['q7'] >= 4 or eval_data['q11'] >= 4) and val_ec <= 2.5:
        insights.append(("Cliente con alto prestigio/passaparola ma basso valore economico", "Massimizzare valore indiretto (case study, referenze), priorità in PR, co-branding."))
    if eval_data['q4'] >= 4:
        insights.append(("Cliente con alti costi logistici", "Razionalizzare processi, rivedere condizioni commerciali (minimi d'ordine), valutare se ridimensionare il rapporto."))
    if np.mean([eval_data[k] for k in ['q7', 'q11', 'q12', 'q13', 'q15']]) >= 4 and sum(eval_data.values()) < 40:
        insights.append(("Cliente con alto potenziale ma basso punteggio attuale", "Inserire in programmi di sviluppo, incontri strategici, proporre nuovi prodotti, usare CRM per obiettivi."))
    if eval_data['q11'] >= 4 and val_rel >= 4:
        insights.append(("Cliente Ambasciatore", "Attivare programma di referral, invitarlo come ospite a eventi, offrirgli anteprime, creare storia di successo congiunta."))
    return insights

# --- INTERFACCIA STREAMLIT ---

col1_title, col2_title = st.columns([1, 10])
logo_path = Path("data/Logo_nyfil.png")
if logo_path.exists():
    col1_title.image(str(logo_path), width=80)
col2_title.title("Dashboard Analisi Clienti")


df_clienti = load_clients_df()
df_ordini = load_all_orders_df()

if df_clienti.empty:
    st.warning(f"File '{CLIENTS_CSV}' non trovato.")
    st.stop()

# --- HOME PAGE ---
anni_disponibili = sorted(df_clienti['ANNO'].unique(), reverse=True)
col1, col2 = st.columns(2)
anni_selezionati = col1.multiselect("Filtra per Anno", options=anni_disponibili, default=anni_disponibili)
paese_selezionato = col2.selectbox("Filtra per Paese", options=["Tutti", "Italia", "Estero"])

st.header("Macrodati Ordini (per anni selezionati)")
if not df_ordini.empty and anni_selezionati:
    ordini_filtrati_globale = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
    col1_macro, col2_macro, col3_macro = st.columns(3)
    with col1_macro:
        st.markdown("###### Top 5 Articoli (per Kg)")
        top_articoli = ordini_filtrati_globale.groupby('ARTICOLO')['KG'].sum().nlargest(5).reset_index()
        st.dataframe(top_articoli, use_container_width=True, hide_index=True)
    with col2_macro:
        st.markdown("###### Top 5 Colori (per Kg)")
        top_colori = ordini_filtrati_globale.groupby('COLORE')['KG'].sum().nlargest(5).reset_index()
        st.dataframe(top_colori, use_container_width=True, hide_index=True)
    with col3_macro:
        st.markdown("###### Top 5 Articolo-Colore (per Kg)")
        top_combinazioni = ordini_filtrati_globale.groupby(['ARTICOLO', 'COLORE'])['KG'].sum().nlargest(5).reset_index()
        st.dataframe(top_combinazioni, use_container_width=True, hide_index=True)
else:
    st.info("Seleziona almeno un anno per visualizzare i macrodati degli ordini.")

st.divider()

df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
if paese_selezionato != "Tutti": df_filtrato = df_filtrato[df_filtrato['PAESE'] == paese_selezionato]

if not df_filtrato.empty:
    total_revenue = df_filtrato['FATTURATO'].sum()
    revenue_italia = df_filtrato[df_filtrato['PAESE'] == 'Italia']['FATTURATO'].sum()
    quota_italia = (revenue_italia / total_revenue * 100) if total_revenue > 0 else 0
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Totale Fatturato (da anagrafica)", format_euro(total_revenue))
    kpi2.metric("Quota Italia", f"{quota_italia:.1f}%")
    kpi3.metric("Quota Estero", f"{100 - quota_italia:.1f}%")
    kpi4.metric("N. Clienti nel filtro", f"{df_filtrato['CLIENTE'].nunique()}")

st.subheader("Ranking Clienti")
if not df_filtrato.empty:
    df_ranking = df_filtrato.groupby('CLIENTE').agg(Fatturato_Anagrafica=('FATTURATO', 'sum'), PAESE=('PAESE', 'first')).reset_index()
    if not df_ordini.empty and anni_selezionati:
        ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
        df_ordini_agg = ordini_filtrati.groupby('nome_cliente').agg(
            KG_Ordinati=('KG', 'sum'),
            Fatturato_Ordini=('FATTURATO_ORDINE', 'sum')
        ).reset_index()
        df_ranking = pd.merge(df_ranking, df_ordini_agg, left_on='CLIENTE', right_on='nome_cliente', how='left')
        df_ranking[['KG_Ordinati', 'Fatturato_Ordini']] = df_ranking[['KG_Ordinati', 'Fatturato_Ordini']].fillna(0)

    else:
        df_ranking['KG_Ordinati'] = 0
        df_ranking['Fatturato_Ordini'] = 0
    
    df_ranking = df_ranking.sort_values('Fatturato_Anagrafica', ascending=False)
    df_display = df_ranking[['CLIENTE', 'PAESE', 'Fatturato_Anagrafica', 'Fatturato_Ordini', 'KG_Ordinati']].copy()
    df_display['Fatturato_Anagrafica'] = df_display['Fatturato_Anagrafica'].apply(format_euro)
    df_display['Fatturato_Ordini'] = df_display['Fatturato_Ordini'].apply(format_euro)
    df_display['KG_Ordinati'] = df_display['KG_Ordinati'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#", "."))
    st.dataframe(df_display, use_container_width=True, hide_index=True)

with st.expander("Segmentazione Clienti (Basata sull'ultimo anno di valutazione)"):
    if not anni_selezionati:
        st.info("Seleziona un anno per visualizzare la segmentazione.")
    else:
        anno_segmentazione = anni_selezionati[0]
        all_evals = []
        for cliente in df_clienti['CLIENTE'].unique():
            eval_data = load_evaluation(cliente, anno_segmentazione)
            if sum(eval_data.values()) != len(eval_data) * 3:
                _, val_ec, val_rel = calculate_scores(eval_data)
                segment = get_matrix_quadrant(val_ec, val_rel)
                ha_valutazione = True
            else:
                segment = 'Valutazione non ancora avvenuta'
                ha_valutazione = False
            all_evals.append({'CLIENTE': cliente, 'VALUTAZIONE': segment, 'Ha_Valutazione': ha_valutazione})
        
        if all_evals:
            df_segments = pd.DataFrame(all_evals)
            df_segments.sort_values(by='Ha_Valutazione', ascending=False, inplace=True)
            st.dataframe(df_segments[['CLIENTE', 'VALUTAZIONE']], use_container_width=True, hide_index=True)
        else:
            st.warning(f"Nessuna valutazione trovata per l'anno {anno_segmentazione}.")

clienti_selezionati = st.multiselect("Seleziona uno o più clienti per visualizzare la scheda di dettaglio", options=df_ranking['CLIENTE'].tolist())

# --- SCHEDA CLIENTE ---
if clienti_selezionati:
    st.divider()
    st.header(f"Scheda Alleati: {', '.join(clienti_selezionati)}")
    anno_riferimento = anni_selezionati[0] if anni_selezionati else anni_disponibili[0]
    tab_eval, tab_dati, tab_ordini = st.tabs(["Valutazione Alleati", "Anagrafica & Fatturato", "Ordini & Statistiche"])
    
    with tab_eval:
        st.subheader("Valutazioni Individuali")
        evals_data = {}
        for cliente in clienti_selezionati:
            with st.expander(f"Valutazione per {cliente} (Anno: {anno_riferimento})"):
                with st.form(key=f"evaluation_form_{cliente}"):
                    eval_data = load_evaluation(cliente, anno_riferimento)
                    
                    cols = st.columns(3)
                    temp_eval_data = {}
                    for i, q in enumerate(EVALUATION_QUESTIONS):
                        with cols[i % 3]:
                            temp_eval_data[q['key']] = st.slider(
                                q['text'], 1, 5, value=eval_data.get(q['key'], 3), key=f"{q['key']}_{cliente}"
                            )
                    
                    submitted = st.form_submit_button("Salva Valutazione")
                    if submitted:
                        save_evaluation(cliente, anno_riferimento, temp_eval_data)
                        st.success(f"Valutazione per {cliente} salvata!")
                        st.cache_data.clear()
                        st.rerun()

                evals_data[cliente] = load_evaluation(cliente, anno_riferimento)


        st.divider()
        st.subheader("Analisi Strategica Comparata")
        
        fig_matrix = go.Figure()
        fig_radar = go.Figure()
        
        for cliente, data in evals_data.items():
            total_score, val_economico, val_relazionale = calculate_scores(data)
            fig_matrix.add_trace(go.Scatter(x=[val_economico], y=[val_relazionale], mode='markers+text', text=cliente, marker=dict(size=15), name=cliente))
            
            radar_values = [data[q['key']] for q in EVALUATION_QUESTIONS]
            fig_radar.add_trace(go.Scatterpolar(r=radar_values + [radar_values[0]], theta=[f"Q{i+1}" for i in range(15)] + ["Q1"], fill='toself', name=cliente, opacity=0.7))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Matrice Decisionale Comparata")
            fig_matrix.update_layout(xaxis_title="Valore Economico", yaxis_title="Valore Relazionale", xaxis=dict(range=[1, 5]), yaxis=dict(range=[1, 5]), shapes=[dict(type='line', x0=3, y0=1, x1=3, y1=5, line=dict(dash='dash')), dict(type='line', x0=1, y0=3, x1=5, y1=3, line=dict(dash='dash'))])
            st.plotly_chart(fig_matrix, use_container_width=True)

        with col2:
            st.markdown("##### Profili Radar Comparati")
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5])))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab_dati:
        for cliente in clienti_selezionati:
            with st.expander(f"Dati per {cliente}"):
                dati_cliente = df_clienti[df_clienti['CLIENTE'] == cliente]
                st.subheader("Anagrafica")
                anagrafica = dati_cliente[dati_cliente['ANNO'] == anno_riferimento].iloc[0]
                cols_anagrafica = st.columns(3)
                cols_anagrafica[0].markdown(f"**Indirizzo:**<br>{anagrafica.get('VIA', 'N/D')}<br>{anagrafica.get('CAP', '')} {anagrafica.get('CITTA', 'N/D')} ({anagrafica.get('PROVINCIA', 'N/D')})", unsafe_allow_html=True)
                cols_anagrafica[1].markdown(f"**Paese:**<br>{anagrafica.get('PAESE', 'N/D')}", unsafe_allow_html=True)
                cols_anagrafica[2].markdown(f"**Contatti:**<br>Titolare: {anagrafica.get('TITOLARE', 'N/D')}<br>Email: {anagrafica.get('EMAIL', 'N/D')}", unsafe_allow_html=True)
                st.divider()
                st.subheader("Andamento Fatturato Annuale (da Anagrafica)")
                fatturato_annuale = dati_cliente.groupby('ANNO')['FATTURATO'].sum().sort_index()
                fig_bar = go.Figure(data=[go.Bar(x=fatturato_annuale.index, y=fatturato_annuale.values, text=[format_euro(v) for v in fatturato_annuale.values], textposition='auto')])
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab_ordini:
        st.subheader("Statistiche Ordini")
        if df_ordini.empty or not anni_selezionati:
            st.info("Seleziona anni e assicurati che i file ordini siano presenti.")
        else:
            ordini_selezionati = df_ordini[(df_ordini['ANNO'].isin(anni_selezionati)) & (df_ordini['nome_cliente'].isin(clienti_selezionati))]
            if ordini_selezionati.empty:
                st.info("Nessun ordine trovato per i clienti selezionati negli anni indicati.")
            else:
                st.subheader("Statistiche Aggregate (Clienti Selezionati)")
                total_kg, total_fatturato_ordini = ordini_selezionati['KG'].sum(), ordini_selezionati['FATTURATO_ORDINE'].sum()
                prezzo_medio_kg = (total_fatturato_ordini / total_kg) if total_kg > 0 else 0
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Kg Totali (Aggregati)", f"{total_kg:,.2f} Kg".replace(",", "."))
                kpi2.metric("Fatturato Ordini (Aggregato)", format_euro(total_fatturato_ordini))
                kpi3.metric("Prezzo Medio Kg (Aggregato)", f"{format_euro(prezzo_medio_kg)} /Kg")
                kpi4.metric("N. Righe Ordine (Aggregate)", f"{len(ordini_selezionati)}")
                st.divider()

                st.subheader("Dettaglio per Cliente")
                for cliente in clienti_selezionati:
                    with st.expander(f"Ordini per {cliente}"):
                        ordini_cliente_singolo = ordini_selezionati[ordini_selezionati['nome_cliente'] == cliente]
                        if ordini_cliente_singolo.empty:
                            st.write("Nessun ordine per questo cliente nel periodo selezionato.")
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
                            # Formattazione colonna Fatturato se esiste
                            if 'Totale_Fatturato' in df_agg.columns:
                                df_display = df_agg.copy()
                                df_display['Totale_Fatturato'] = df_display['Totale_Fatturato'].apply(format_euro)
                                st.dataframe(df_display, use_container_width=True, hide_index=True)
                            else:
                                st.dataframe(df_agg, use_container_width=True, hide_index=True)
                            
                            csv = df_agg.to_csv(index=False, sep=';', decimal=',', encoding='latin1')
                            st.download_button(f"📥 Export {title}", csv, f"{filename_prefix}_{cliente}.csv", "text/csv", key=f"btn_{filename_prefix}_{key_suffix}")

                        # Aggiunta colonna FATTURATO_ORDINE alle aggregazioni
                        agg_articolo_full = ordini_cliente_singolo.groupby('ARTICOLO').agg(
                            Totale_Kg=('KG', 'sum'),
                            Totale_Fatturato=('FATTURATO_ORDINE', 'sum')
                        ).reset_index().sort_values('Totale_Kg', ascending=False)
                        display_agg_table(agg_articolo_full, "Dettaglio Analisi per Articolo", "analisi_articolo", cliente)

                        agg_colore_full = ordini_cliente_singolo.groupby('COLORE').agg(
                            Totale_Kg=('KG', 'sum'),
                            Totale_Fatturato=('FATTURATO_ORDINE', 'sum')
                        ).reset_index().sort_values('Totale_Kg', ascending=False)
                        display_agg_table(agg_colore_full, "Dettaglio Analisi per Colore", "analisi_colore", cliente)

                        agg_articolo_colore_full = ordini_cliente_singolo.groupby(['ARTICOLO', 'COLORE']).agg(
                            Totale_Kg=('KG', 'sum'),
                            Totale_Fatturato=('FATTURATO_ORDINE', 'sum')
                        ).reset_index().sort_values('Totale_Kg', ascending=False)
                        display_agg_table(agg_articolo_colore_full, "Dettaglio Analisi per Articolo e Colore", "analisi_articolo_colore", cliente)
