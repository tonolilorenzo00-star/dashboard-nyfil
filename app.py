import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import numpy as np
import glob
import re
import json
import hashlib
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

# --- CONFIGURAZIONE PAGINA E COSTANTI ---
st.set_page_config(page_title="Analisi Clienti Nyfil", layout="wide")

# Percorsi
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "app.db"
CLIENTS_CSV = DATA_DIR / "elenco clienti.csv"
DATA_DIR.mkdir(exist_ok=True)

# Province e regioni
PROVINCE_ITALIANE = [
    "AG","AL","AN","AO","AR","AP","AT","AV","BA","BT","BL","BN","BG","BI","BO","BZ","BS","CA","CL","CB","CE","CH","CO","CS","CR","KR","CN","EN","FE","FI","FG","FC","FR","GE","GO","GR","IM","IS","SP","AQ","LT","LE","LC","LI","LO","LU","MC","MN","MS","MT","ME","MI","MO","MB","NA","NO","NU","OR","PD","PA","PR","PV","PG","PU","PE","PC","PI","PT","PN","PZ","PO","RG","RA","RC","RE","RI","RN","RM","RO","SA","SS","SV","SI","SO","SR","TA","TE","TR","TO","TP","TN","TV","TS","UD","VA","VE","VB","VC","VR","VV","VI","VT"
]
REGIONE_TO_PROV = {
    "veneto": {"VR","VI","VE","PD","TV","BL","RO"},
    "lombardia": {"MI","MB","BG","BS","CO","CR","LC","LO","MN","PV","SO","VA"},
    "piemonte": {"AL","AT","BI","CN","NO","TO","VB","VC"},
    "emilia-romagna": {"BO","FE","FC","MO","PR","PC","RA","RE","RN"},
    "toscana": {"AR","FI","GR","LI","LU","MS","PI","PO","PT","SI"},
    "lazio": {"FR","LT","RI","RM","VT"},
}

EVALUATION_QUESTIONS = [
    {"key":"q1","text":"1. GENERA FATTURATI IMPORTANTI PER NOI?","category":"Valore Economico"},
    {"key":"q2","text":"2. DIMOSTRA INTERESSE E COMPRENDE L’UTILITÀ DEL NS. PRODOTTO/SERVIZIO?","category":"Valore Relazionale"},
    {"key":"q3","text":"3. SIAMO IMPORTANTI PER LUI, CI VALUTA POSITIVAMENTE, TIENE A NOI?","category":"Valore Relazionale"},
    {"key":"q4","text":"4. RICHIEDE SFORZI E COSTI LOGISTICI?","category":"Costi e Sforzi"},
    {"key":"q5","text":"5. SI E' CREATO UN BUON LIVELLO DI EMPATIA?","category":"Valore Relazionale"},
    {"key":"q6","text":"6. E' SODDISFATTO DEL VALORE CHE RIUSCIAMO AD ASSICURARGLI?","category":"Valore Relazionale"},
    {"key":"q7","text":"7. AGGIUNGE PRESTIGIO AL NOSTRO PARCO ALLEATI?","category":"Potenziale Futuro"},
    {"key":"q8","text":"8. ABBIAMO CON LUI UN RAPPORTO DURATURO NEL TEMPO?","category":"Valore Relazionale"},
    {"key":"q9","text":"9. HA UNA BUONA FREQUENZA DI LAVORI?","category":"Valore Economico"},
    {"key":"q10","text":"10. RIUSCIAMO A GENERARE SODDISFAZIONE?","category":"Valore Relazionale"},
    {"key":"q11","text":"11. CI AIUTA AD ACQUISIRE NUOVI CLIENTI TRAMITE PASSAPAROLA?","category":"Potenziale Futuro"},
    {"key":"q12","text":"12. LE SUE CARATTERISTICHE LO RENDONO UN ALLEATO SVILUPPABILE?","category":"Potenziale Futuro"},
    {"key":"q13","text":"13. HA CLIENTI CON BUON POTENZIALE ECONOMICO?","category":"Valore Economico"},
    {"key":"q14","text":"14. ABBIAMO UNA BUONA LEADERSHIP NEI SUOI CONFRONTI?","category":"Valore Relazionale"},
    {"key":"q15","text":"15. E' UN PARTNER CON IL QUALE POTER CONDIVIDERE LA NOSTRA STRATEGIA?","category":"Potenziale Futuro"},
]

# ------------------ UTILS DATI ------------------
def format_euro_robust(value):
    try:
        if pd.isna(value) or not isinstance(value,(int,float)):
            return "N/A"
        return f"€ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "N/A"

def parse_decimal_string(s: str) -> float:
    if not isinstance(s, str): s = str(s)
    try: return float(s.replace('.','').replace(',', '.'))
    except Exception: return 0.0

def normalize_year(s: str) -> str:
    s = str(s).strip()
    if s.isdigit():
        if len(s)==2: return f"20{s}"
        if len(s)==4: return s
    return s

def clean_customer_name(name: str) -> str:
    if not isinstance(name, str): return ""
    return re.sub(r'\s+',' ',name).strip().lower()

def detect_country(cap: str, provincia: str) -> str:
    cap_str, prov_str = str(cap).strip(), str(provincia).strip().upper()
    if not prov_str or prov_str=='NAN': return "Estero"
    if (cap_str.isdigit() and len(cap_str)==5) or prov_str in PROVINCE_ITALIANE: return "Italia"
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cliente TEXT, anno_rif TEXT,
                digest TEXT, output TEXT,
                created_at TEXT
            );
        """)
        # NEW: tabella per stelle, priorità, note e stato azione
        conn.execute("""
            CREATE TABLE IF NOT EXISTS client_actions (
                cliente TEXT NOT NULL,
                anno_rif TEXT NOT NULL,
                stars INTEGER DEFAULT 0,
                priority REAL DEFAULT 0,
                note TEXT,
                done INTEGER DEFAULT 0,
                updated_at TEXT,
                PRIMARY KEY (cliente, anno_rif)
            );
        """)

@st.cache_data
def load_clients_df(uploaded_file=None) -> pd.DataFrame:
    source = CLIENTS_CSV if CLIENTS_CSV.exists() else uploaded_file
    if not source: return pd.DataFrame()
    try:
        df = pd.read_csv(source, sep=';', encoding='latin1', low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        required_cols = {
            "nome_clietne":"CLIENTE","nome_cliente":"CLIENTE","via":"VIA","città":"CITTA",
            "cap":"CAP","provincia":"PROVINCIA","titolare_azienda":"TITOLARE",
            "recapiti_mail":"EMAIL","anno":"ANNO_ORIG","imponibile":"IMPONIBILE"
        }
        df.rename(columns=required_cols, inplace=True)
        if 'CLIENTE' in df.columns:
            df['CLIENTE'] = df['CLIENTE'].apply(clean_customer_name)
        df['ANNO_ORIG'] = df['ANNO_ORIG'].astype(str).str.split('/')
        df = df.explode('ANNO_ORIG')
        df['ANNO'] = df['ANNO_ORIG'].apply(normalize_year)
        df['FATTURATO'] = df['IMPONIBILE'].apply(parse_decimal_string)
        df['PAESE'] = df.apply(lambda r: detect_country(r.get('CAP',''), r.get('PROVINCIA','')), axis=1)
        agg_cols = {k:'first' for k in ['VIA','CITTA','CAP','PROVINCIA','TITOLARE','EMAIL','PAESE']}
        agg_cols['FATTURATO'] = 'sum'
        cols_to_agg = {k:v for k,v in agg_cols.items() if k in df.columns}
        return df.groupby(['CLIENTE','ANNO']).agg(cols_to_agg).reset_index()
    except Exception as e:
        st.error(f"Errore lettura file clienti: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all_orders_df() -> pd.DataFrame:
    order_files = glob.glob(str(DATA_DIR / "ordini_*"))
    if not order_files: return pd.DataFrame()
    dfs = []
    for file in order_files:
        try:
            year_match = re.search(r'(\d+)', Path(file).stem)
            if not year_match: continue
            year_full = normalize_year(year_match.group(1))
            df = pd.read_csv(file, sep=';', encoding='latin1', low_memory=False) if file.endswith('.csv') else pd.read_excel(file)
            df.columns = [re.sub(r'_\d+$','',c).strip().lower() for c in df.columns]
            df['ANNO'] = year_full
            dfs.append(df)
        except Exception as e:
            st.warning(f"Impossibile leggere {file}: {e}")
    if not dfs: return pd.DataFrame()
    df_orders = pd.concat(dfs, ignore_index=True)
    if 'nome_cliente' in df_orders.columns:
        df_orders['nome_cliente'] = df_orders['nome_cliente'].apply(clean_customer_name)
    df_orders.dropna(subset=['articolo_colore','quantita'], inplace=True)
    df_orders = df_orders[df_orders['quantita'] != 0].copy()
    df_orders['FATTURATO_ORDINE'] = df_orders['imponibile'].apply(parse_decimal_string) if 'imponibile' in df_orders.columns else 0
    df_orders['KG'] = df_orders['quantita'].apply(parse_decimal_string).round(2)
    split_data = df_orders['articolo_colore'].str.rsplit(' - ', n=1, expand=True)
    df_orders['ARTICOLO'] = split_data[0].str.strip()
    df_orders['COLORE'] = split_data[1].str.strip().fillna('NON SPECIFICATO')
    return df_orders[['nome_cliente','ANNO','ARTICOLO','COLORE','KG','FATTURATO_ORDINE']]

def load_evaluation(cliente: str, anno: str) -> dict:
    conn = get_db_connection()
    cur = conn.cursor()
    keys = [q['key'] for q in EVALUATION_QUESTIONS]
    cur.execute(f"SELECT {', '.join(keys)} FROM evaluation WHERE cliente=? AND anno=?", (cliente, anno))
    row = cur.fetchone()
    if row: return dict(zip(keys, row))
    return {k:3 for k in keys}

def save_evaluation(cliente: str, anno: str, data_dict: dict):
    try:
        conn = get_db_connection()
        keys = [q['key'] for q in EVALUATION_QUESTIONS]
        cols = ", ".join(keys)
        placeholders = ", ".join(["?"]*len(keys))
        q = f"INSERT OR REPLACE INTO evaluation (cliente, anno, {cols}, updated_at) VALUES (?, ?, {placeholders}, ?)"
        vals = [cliente, anno] + [data_dict.get(k,1) for k in keys] + [datetime.now().isoformat()]
        with conn:
            conn.execute(q, tuple(vals))
        st.toast(f"Valutazione per {cliente.upper()} ({anno}) salvata!")
    except sqlite3.OperationalError as e:
        st.error(f"Errore di salvataggio: {e}.")

def calculate_scores(eval_data):
    val_economico_keys = [q['key'] for q in EVALUATION_QUESTIONS if q['category']=="Valore Economico"]
    val_relazionale_keys = [q['key'] for q in EVALUATION_QUESTIONS if q['category']=="Valore Relazionale"]
    total_score = sum(eval_data.values())
    val_economico = np.mean([eval_data[k] for k in val_economico_keys])
    val_relazionale = np.mean([eval_data[k] for k in val_relazionale_keys])
    return total_score, val_economico, val_relazionale

def get_matrix_quadrant(x,y):
    if x>3 and y>3: return "Partner Chiave"
    if x>3 and y<=3: return "Specialista Redditizio"
    if x<=3 and y>3: return "Amico a Basso Impatto"
    return "Cliente Marginale"

# ------------------ LLM UTILS ------------------
@st.cache_resource
def get_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        return None
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MODEL_NAME = st.secrets.get("MODEL_NAME","gpt-4o-mini")
TEMPERATURE = float(st.secrets.get("MODEL_TEMPERATURE",0.2))
ANON_SALT = st.secrets.get("ANON_SALT","nyfil")

def _truncate(s: str, max_chars: int = 15000) -> str:
    s = str(s)
    return s if len(s)<=max_chars else s[:max_chars] + "\n... [troncato]"

@retry(wait=wait_random_exponential(min=1, max=6), stop=stop_after_attempt(3))
def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = get_openai_client()
    if client is None:
        return "⚠️ Configura OPENAI_API_KEY nei secrets."
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":_truncate(user_prompt)}
        ]
    )
    return resp.choices[0].message.content.strip()

def anon_client_id(name: str) -> str:
    h = hashlib.sha256((ANON_SALT + (name or "")).encode("utf-8")).hexdigest()[:8].upper()
    return f"CLIENTE_{h}"

def build_client_digest_anon(cliente, anni_sel, anno_rif, dfc, dfo):
    alias = anon_client_id(cliente)
    anni_sel = [str(a) for a in anni_sel] if anni_sel else []
    eval_data = load_evaluation(cliente, anno_rif)
    tot, val_ec, val_rel = calculate_scores(eval_data)
    fatt_cli = (dfc[(dfc['CLIENTE']==cliente)&(dfc['ANNO'].isin(anni_sel))]
                .groupby('ANNO',as_index=False)['FATTURATO'].sum().sort_values('ANNO'))
    ord_cli = dfo[(dfo['nome_cliente']==cliente) & (dfo['ANNO'].isin(anni_sel))]
    kg_tot = float(ord_cli['KG'].sum()); fatt_o = float(ord_cli['FATTURATO_ORDINE'].sum())
    prezzo = (fatt_o/kg_tot) if kg_tot>0 else 0
    lines=[]
    lines.append(f"Cliente (anonimo): {alias} | Anno valutazione: {anno_rif} | Anni: {', '.join(anni_sel) or 'tutti'}")
    lines.append(f"Valutazione → Tot: {tot:.1f} | Econ: {val_ec:.2f} | Rel: {val_rel:.2f}")
    lines.append(f"Fatturato per anno (valori €): { {r['ANNO']: float(r['FATTURATO']) for _,r in fatt_cli.iterrows()} }")
    lines.append(f"Ordini → Kg: {kg_tot:.2f} | Fatt: {fatt_o:.2f} | €/Kg: {prezzo:.3f}")
    return "\n".join(lines)

def save_ai_output(cliente: str, anno_rif: str, digest: str, output: str):
    try:
        conn = get_db_connection()
        with conn:
            conn.execute(
                "INSERT INTO ai_insights (cliente, anno_rif, digest, output, created_at) VALUES (?, ?, ?, ?, ?)",
                (cliente, anno_rif, digest, output, datetime.now().isoformat())
            )
    except sqlite3.OperationalError:
        pass

def load_last_ai_output(cliente: str, anno_rif: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT output, created_at FROM ai_insights WHERE cliente=? AND anno_rif=? ORDER BY id DESC LIMIT 1",
        (cliente, anno_rif)
    )
    row = cur.fetchone()
    if row:
        return {"output": row[0], "created_at": row[1]}
    return None

# ---------- ACTIONS: stelle, priorità, note, done ----------
def stars_from_total(total_score: float) -> int:
    # 15 domande * max 5 = 75
    if total_score >= 65: return 5
    if total_score >= 55: return 4
    if total_score >= 45: return 3
    if total_score >= 35: return 2
    return 1

def priority_from_quadrant(val_ec: float, val_rel: float) -> float:
    quad = get_matrix_quadrant(val_ec, val_rel)
    # più alto => più urgente
    if quad == "Specialista Redditizio": base = 5.0
    elif quad == "Amico a Basso Impatto": base = 4.0
    elif quad == "Partner Chiave": base = 2.0
    else: base = 3.0  # Cliente Marginale
    # penalizza relazionale alto (già ok), aumenta se relazionale basso
    adj = (5 - val_rel) * 0.5 + (5 - min(val_ec,5)) * 0.3
    return round(base + adj, 2)

def upsert_client_action(cliente: str, anno_rif: str, stars: int = None,
                         priority: float = None, note: str = None, done: int = None):
    conn = get_db_connection()
    now = datetime.now().isoformat()
    with conn:
        # se esiste riga, aggiorna; altrimenti crea
        cur = conn.execute("SELECT 1 FROM client_actions WHERE cliente=? AND anno_rif=?", (cliente, anno_rif))
        exists = cur.fetchone() is not None
        if not exists:
            conn.execute(
                "INSERT INTO client_actions (cliente, anno_rif, stars, priority, note, done, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cliente, anno_rif, stars or 0, priority or 0.0, note or "", done or 0, now)
            )
        else:
            sets = []; vals = []
            if stars is not None: sets.append("stars=?"); vals.append(int(stars))
            if priority is not None: sets.append("priority=?"); vals.append(float(priority))
            if note is not None: sets.append("note=?"); vals.append(note)
            if done is not None: sets.append("done=?"); vals.append(int(done))
            sets.append("updated_at=?"); vals.append(now)
            if sets:
                sql = f"UPDATE client_actions SET {', '.join(sets)} WHERE cliente=? AND anno_rif=?"
                vals += [cliente, anno_rif]
                conn.execute(sql, tuple(vals))

def load_actions_table(anno_rif: str):
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM client_actions WHERE anno_rif = ?", conn, params=[anno_rif])
    return df

# ---------- Nuovi clienti per anno ----------
@st.cache_data
def compute_first_year_per_client(df_clienti: pd.DataFrame, df_ordini: pd.DataFrame) -> pd.DataFrame:
    # primo anno in anagrafica
    first_a = (df_clienti.groupby('CLIENTE')['ANNO'].min().reset_index().rename(columns={'ANNO':'FIRST_YEAR'}))
    # se vuoi includere anche primo anno ordini (più robusto)
    if not df_ordini.empty:
        first_o = (df_ordini.groupby('nome_cliente')['ANNO'].min().reset_index().rename(columns={'nome_cliente':'CLIENTE','ANNO':'FIRST_YEAR_ORD'}))
        first = pd.merge(first_a, first_o, on='CLIENTE', how='outer')
        first['FIRST_YEAR'] = first[['FIRST_YEAR','FIRST_YEAR_ORD']].min(axis=1)
        first.drop(columns=['FIRST_YEAR_ORD'], inplace=True)
    else:
        first = first_a
    return first  # colonne: CLIENTE, FIRST_YEAR

# ------------------ PARSER NATURALE & COPILOT (come già impostato) ------------------
# (omessi per brevità: se già hai le versioni avanzate che ti ho passato, lasciale così)
# ---- In questo file completo, tieni il tuo blocco Copilot esistente ----

# ------------------ PAGINE APP ------------------
def page_dashboard(df_clienti, df_ordini, anni_selezionati, paese_selezionato, analysis_mode):
    st.title("Dashboard Riepilogativa")
    df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
    if paese_selezionato != "Tutti":
        df_filtrato = df_filtrato[df_filtrato['PAESE']==paese_selezionato]

    st.header("KPI Generali (da anagrafica)")
    if not df_filtrato.empty:
        total_revenue = df_filtrato['FATTURATO'].sum()
        revenue_italia = df_filtrato[df_filtrato['PAESE']=='Italia']['FATTURATO'].sum()
        quota_italia = (revenue_italia/total_revenue*100) if total_revenue>0 else 0
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Totale Fatturato", format_euro_robust(total_revenue))
        k2.metric("Quota Italia", f"{quota_italia:.1f}%")
        k3.metric("Quota Estero", f"{100-quota_italia:.1f}%")
        k4.metric("N. Clienti nel filtro", f"{df_filtrato['CLIENTE'].nunique()}")
    else:
        st.info("Nessun dato anagrafico per i filtri selezionati.")

    st.header("Macrodati Ordini")
    if not df_ordini.empty and anni_selezionati:
        ordini_filtrati_globale = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
        if analysis_mode == "Aggrega Anni":
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown("###### Top 10 Articoli (per Kg)")
                st.dataframe(ordini_filtrati_globale.groupby('ARTICOLO')['KG'].sum().nlargest(10).reset_index(),
                             use_container_width=True, hide_index=True, height=385)
            with c2:
                st.markdown("###### Top 10 Colori (per Kg)")
                st.dataframe(ordini_filtrati_globale.groupby('COLORE')['KG'].sum().nlargest(10).reset_index(),
                             use_container_width=True, hide_index=True, height=385)
            with c3:
                st.markdown("###### Top 10 Articolo-Colore (per Kg)")
                st.dataframe(ordini_filtrati_globale.groupby(['ARTICOLO','COLORE'])['KG'].sum().nlargest(10).reset_index(),
                             use_container_width=True, hide_index=True, height=385)
        else:
            st.info("Modalità Confronto Anni: andamento dei 10 articoli più importanti nel periodo.")
            top_10 = ordini_filtrati_globale.groupby('ARTICOLO')['KG'].sum().nlargest(10).index
            dfc = ordini_filtrati_globale[ordini_filtrati_globale['ARTICOLO'].isin(top_10)]
            pivot = dfc.pivot_table(index='ARTICOLO', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
            pivot['Totale'] = pivot.sum(axis=1)
            pivot.sort_values('Totale', ascending=False, inplace=True)
            del pivot['Totale']
            st.dataframe(pivot.style.format("{:,.2f} Kg"), use_container_width=True)
    else:
        st.info("Seleziona almeno un anno per visualizzare i macrodati degli ordini.")

def page_elenco_clienti(df_clienti, df_ordini, anni_selezionati, paese_selezionato, analysis_mode):
    st.title("Elenco e Segmentazione Clienti")
    df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
    if paese_selezionato != "Tutti":
        df_filtrato = df_filtrato[df_filtrato['PAESE']==paese_selezionato]
    st.subheader("Ranking Clienti")
    if not df_filtrato.empty:
        df_ranking_base = df_filtrato.groupby('CLIENTE').agg(PAESE=('PAESE','first')).reset_index()
        if analysis_mode == "Aggrega Anni":
            df_ranking = df_filtrato.groupby('CLIENTE').agg(Fatturato_Anagrafica=('FATTURATO','sum')).reset_index()
            if not df_ordini.empty and anni_selezionati:
                ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
                df_ord_agg = ordini_filtrati.groupby('nome_cliente').agg(KG_Ordinati=('KG','sum')).reset_index()
                df_ranking = pd.merge(df_ranking, df_ord_agg, left_on='CLIENTE', right_on='nome_cliente', how='left')
            df_ranking = pd.merge(df_ranking, df_ranking_base, on='CLIENTE', how='left')
            df_ranking['KG_Ordinati'] = df_ranking['KG_Ordinati'].fillna(0)
            df_ranking = df_ranking.sort_values('Fatturato_Anagrafica', ascending=False)
            df_ranking['CLIENTE_DISPLAY'] = df_ranking['CLIENTE'].str.upper()
            df_display = df_ranking[['CLIENTE_DISPLAY','PAESE','Fatturato_Anagrafica','KG_Ordinati']].copy()
            df_display.rename(columns={'CLIENTE_DISPLAY':'CLIENTE'}, inplace=True)
            df_display['Fatturato_Anagrafica'] = df_display['Fatturato_Anagrafica'].apply(format_euro_robust)
            df_display['KG_Ordinati'] = df_display['KG_Ordinati'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#","."))
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("Modalità Confronto Anni: le tabelle mostrano i dati disaggregati per anno.")
            fatt_pivot = df_filtrato.pivot_table(index='CLIENTE', columns='ANNO', values='FATTURATO', aggfunc='sum')
            ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
            kg_pivot = ordini_filtrati.pivot_table(index='nome_cliente', columns='ANNO', values='KG', aggfunc='sum')
            df_compare = pd.merge(fatt_pivot, kg_pivot, left_index=True, right_index=True, how='outer').fillna(0)
            col_order = []
            for year in sorted(anni_selezionati, reverse=True):
                cf, ck = f"Fatturato_{year}", f"KG_{year}"
                df_compare.rename(columns={year+'_x':cf, year+'_y':ck}, inplace=True, errors='ignore')
                if cf in df_compare.columns: df_compare[cf] = df_compare[cf].apply(format_euro_robust)
                if ck in df_compare.columns: df_compare[ck] = df_compare[ck].apply(lambda x: f"{x:,.2f} Kg")
                col_order.extend([c for c in [cf,ck] if c in df_compare.columns])
            df_compare = pd.merge(df_compare, df_ranking_base, left_index=True, right_on='CLIENTE', how='left').set_index('CLIENTE')
            df_compare.reset_index(inplace=True)
            df_compare['CLIENTE'] = df_compare['CLIENTE'].str.upper()
            st.dataframe(df_compare[['CLIENTE','PAESE']+col_order], use_container_width=True, hide_index=True)

# --- REPORT AVANZATI (già presente) ---
def page_report_avanzati(df_ordini):
    st.title("Report Avanzati — Analisi Predittiva per Articolo")
    if df_ordini.empty or 'ARTICOLO' not in df_ordini.columns:
        st.info("Nessun dato ordini disponibile o colonna 'ARTICOLO' assente."); return
    articoli = sorted([a for a in df_ordini['ARTICOLO'].dropna().unique() if str(a).strip()!=""])
    if not articoli: st.info("Nessun articolo disponibile."); return
    articolo_sel = st.selectbox("Seleziona un articolo", options=articoli, index=0)
    df_art = (df_ordini[df_ordini['ARTICOLO']==articolo_sel]
              .assign(ANNO_NUM=pd.to_numeric(df_ordini['ANNO'], errors='coerce'))
              .dropna(subset=['ANNO_NUM'])
              .groupby('ANNO_NUM', as_index=False)['KG'].sum()
              .sort_values('ANNO_NUM'))
    if df_art.empty: st.warning("Nessun dato storico per questo articolo."); return
    x = df_art['ANNO_NUM'].values.astype(float); y = df_art['KG'].values.astype(float)
    if len(df_art)>=2:
        m,q = np.polyfit(x,y,1); x_line = np.linspace(x.min(), x.max()+1, 100); y_line = m*x_line + q
        slope_desc = "in crescita" if m>0 else ("in calo" if m<0 else "stabile")
    else:
        m,q = 0, y[0]; x_line, y_line = x, y; slope_desc = "dati insufficienti per trend"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_art['ANNO_NUM'], y=df_art['KG'], mode='markers+lines', name='Storico Kg'))
    if len(df_art)>=2:
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Trend (regressione lineare)', line=dict(dash='dash')))
        next_year = int(np.max(x))+1; y_next = float(m*next_year + q)
        fig.add_trace(go.Scatter(x=[next_year], y=[y_next], mode='markers', name='Proiezione anno +1',
                                 marker=dict(symbol='diamond-open', size=10)))
    fig.update_layout(title=f"Andamento Kg per Articolo: {articolo_sel}", xaxis_title="Anno", yaxis_title="Kg", hovermode="x unified")
    a,b,c = st.columns(3)
    a.metric("Anni coperti", f"{df_art['ANNO_NUM'].nunique()}"); b.metric("Kg totali", f"{df_art['KG'].sum():,.2f}".replace(",", ".")); c.metric("Tendenza", slope_desc.capitalize())
    st.plotly_chart(fig, use_container_width=True)
    if len(df_art)>=2: st.caption("Linea tratteggiata = regressione sui dati storici; rombo = proiezione anno successivo.")

# --- ANALISI DETTAGLIATA (aggiornata con filtro Nuovi e Stelle/Azioni) ---
def page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode):
    st.title("Analisi Dettagliata Cliente")

    # Calcolo "nuovi clienti per anno"
    df_first = compute_first_year_per_client(df_clienti, df_ordini)  # CLIENTE, FIRST_YEAR

    # Selettori: filtro "Nuovi [anno]" e scelta clienti
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        anno_new = st.selectbox("Anno 'nuovi clienti'", options=anni_disponibili, index=0, key="anno_nuovi")
    with c2:
        only_new = st.checkbox(f"Mostra solo nuovi clienti {anno_new}", value=False, key="flag_nuovi")
    with c3:
        # costruiamo la lista opzioni
        clienti_all = df_clienti['CLIENTE'].unique().tolist()
        if only_new:
            new_set = set(df_first[df_first['FIRST_YEAR']==str(anno_new)]['CLIENTE'])
            clienti_options = sorted([c.upper() for c in clienti_all if c in new_set])
        else:
            clienti_options = sorted([c.upper() for c in clienti_all])
        clienti_selezionati_upper = st.multiselect("Seleziona uno o più clienti", options=clienti_options, key='client_selector_detail')

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

    # === TABELLE AZIONI SOTTO IL SELETTORE ===
    st.subheader("📌 Azioni Prioritarie")
    anno_rif_tab = anni_scheda_selezionati[0] if anni_scheda_selezionati else anni_disponibili[0]
    # carica azioni per l'anno di riferimento
    df_actions = load_actions_table(anno_rif_tab)
    # arricchisco con display name
    if not df_actions.empty:
        df_actions['CLIENTE_DISPLAY'] = df_actions['cliente'].str.upper()
        todo = df_actions[df_actions['done']==0].copy()
        done = df_actions[df_actions['done']==1].copy()

        # ordina per priorità (desc)
        todo.sort_values(by=['priority','updated_at'], ascending=[False, False], inplace=True)
        done.sort_values(by=['updated_at'], ascending=False, inplace=True)

        col_todo, col_done = st.columns(2)
        with col_todo:
            st.markdown("**Da fare (ordinate per priorità)**")
            if todo.empty:
                st.info("Nessuna azione in sospeso per questo anno.")
            else:
                for _, row in todo.iterrows():
                    with st.expander(f"{row['CLIENTE_DISPLAY']} — ⭐ {int(row['stars'])} — Priorità {row['priority']:.2f}"):
                        note_val = st.text_area("Nota/Azione", value=row.get('note',''), key=f"note_todo_{row['cliente']}_{anno_rif_tab}")
                        done_ck = st.checkbox("Segna come fatto", value=False, key=f"done_todo_{row['cliente']}_{anno_rif_tab}")
                        if st.button("Salva", key=f"save_todo_{row['cliente']}_{anno_rif_tab}"):
                            upsert_client_action(row['cliente'], anno_rif_tab, note=note_val, done=1 if done_ck else 0)
                            st.success("Aggiornato.")
                            st.experimental_rerun()

        with col_done:
            st.markdown("**Completate**")
            if done.empty:
                st.info("Nessuna azione completata.")
            else:
                for _, row in done.iterrows():
                    with st.expander(f"{row['CLIENTE_DISPLAY']} — ⭐ {int(row['stars'])} — Priorità {row['priority']:.2f}"):
                        note_val = st.text_area("Nota", value=row.get('note',''), key=f"note_done_{row['cliente']}_{anno_rif_tab}")
                        undo_ck = st.checkbox("Riporta in 'Da fare'", value=False, key=f"undo_done_{row['cliente']}_{anno_rif_tab}")
                        if st.button("Aggiorna", key=f"save_done_{row['cliente']}_{anno_rif_tab}"):
                            upsert_client_action(row['cliente'], anno_rif_tab, note=note_val, done=0 if undo_ck else 1)
                            st.success("Aggiornato.")
                            st.experimental_rerun()
    else:
        st.info("Nessuna azione salvata per quest'anno. Verranno create quando fai l'analisi IA/Swot.")

    # === Schede Cliente ===
    st.header(f"Scheda Alleati: {', '.join(clienti_selezionati_upper)}")
    anno_riferimento_scheda = anni_scheda_selezionati[0] if anni_scheda_selezionati else anni_disponibili[0]
    tab_eval, tab_dati, tab_ordini = st.tabs(["Valutazione & IA","Anagrafica & Fatturato","Ordini & Statistiche"])

    with tab_eval:
        st.subheader(f"Valutazioni + Azioni (Anno di riferimento: {anno_riferimento_scheda})")
        for cliente in clienti_selezionati:
            with st.expander(f"{cliente.upper()}"):
                # --- Valutazione slider ---
                with st.form(key=f"evaluation_form_{cliente}_{anno_riferimento_scheda}"):
                    ev = load_evaluation(cliente, anno_riferimento_scheda)
                    cols = st.columns(3); temp={}
                    for i,q in enumerate(EVALUATION_QUESTIONS):
                        with cols[i%3]:
                            temp[q['key']] = st.slider(q['text'],1,5,value=ev.get(q['key'],3), key=f"{q['key']}_{cliente}_{anno_riferimento_scheda}")
                    submitted = st.form_submit_button("Salva Valutazione")
                    if submitted:
                        save_evaluation(cliente, anno_riferimento_scheda, temp)
                        st.cache_data.clear()
                        st.rerun()

                # Punteggi e stelle / priorità
                ev_cur = load_evaluation(cliente, anno_riferimento_scheda)
                tot, val_ec, val_rel = calculate_scores(ev_cur)
                quad = get_matrix_quadrant(val_ec, val_rel)
                auto_stars = stars_from_total(tot)
                auto_priority = priority_from_quadrant(val_ec, val_rel)

                st.markdown(f"**Totale**: {tot:.0f}/75 • **Val.Economico**: {val_ec:.2f} • **Val.Relazionale**: {val_rel:.2f} • **Profilo**: _{quad}_")
                cst1, cst2, cst3 = st.columns([1,1,2])
                with cst1:
                    stars_sel = st.selectbox("Stelle (1–5)", options=[1,2,3,4,5], index=auto_stars-1, key=f"stars_{cliente}_{anno_riferimento_scheda}")
                with cst2:
                    st.metric("Priorità (auto)", f"{auto_priority:.2f}")
                with cst3:
                    note_act = st.text_input("Nota/Azione da intraprendere", key=f"note_{cliente}_{anno_riferimento_scheda}", placeholder="Es. programmare visita, proposta up-sell, ecc.")

                # Ultima analisi IA salvata
                saved = load_last_ai_output(cliente, anno_riferimento_scheda)
                if saved:
                    st.markdown("**Ultima analisi IA salvata**")
                    st.caption(f"Generata il: {saved['created_at']}")
                    st.markdown(saved['output'])
                else:
                    st.info("Nessuna analisi IA salvata per questo cliente/anno.")

                # Bottoni: Analizza con IA + Salva Stella/Azione
                colb1, colb2 = st.columns([1,1])
                with colb1:
                    if st.button("🔎 Analizza con IA", key=f"ai_analyze_{cliente}_{anno_riferimento_scheda}"):
                        with st.spinner("Analisi IA in corso..."):
                            digest = build_client_digest_anon(cliente, anni_scheda_selezionati, anno_riferimento_scheda, df_clienti, df_ordini)
                            system = ("Sei un business analyst per un'azienda B2B (nylon). Usa SOLO i dati forniti. "
                                      "Output in italiano, conciso, a punti: 1) Sintesi 2) Opportunità 3) Rischi 4) Azioni 30/60/90 gg.")
                            user = f"DATI CLIENTE (digest anonimizzato):\n{digest}"
                            out = call_llm(system, user)
                            st.markdown("#### Risultato IA"); st.markdown(out)
                            save_ai_output(cliente, anno_riferimento_scheda, digest, out)
                            # allineo/creo riga azione con auto_stars & auto_priority
                            upsert_client_action(cliente, anno_riferimento_scheda, stars=auto_stars, priority=auto_priority)
                            st.experimental_rerun()
                with colb2:
                    if st.button("💾 Salva Stelle/Priorità/Azione", key=f"save_action_{cliente}_{anno_riferimento_scheda}"):
                        upsert_client_action(cliente, anno_riferimento_scheda, stars=stars_sel, priority=auto_priority, note=note_act, done=0)
                        st.success("Azione salvata/aggiornata. La voce compare nella tabella 'Da fare' in alto.")
                        st.experimental_rerun()

    with tab_dati:
        for cliente in clienti_selezionati:
            with st.expander(f"Dati per {cliente.upper()}"):
                dati_cliente = df_clienti[df_clienti['CLIENTE']==cliente]
                st.subheader(f"Anagrafica (Riferimento anno: {anno_riferimento_scheda})")
                anag_anno = dati_cliente[dati_cliente['ANNO']==anno_riferimento_scheda]
                if not anag_anno.empty: anagrafica = anag_anno.iloc[0]
                elif not dati_cliente.empty:
                    anagrafica = dati_cliente.sort_values('ANNO', ascending=False).iloc[0]
                    st.info(f"Dati anagrafici per l'anno {anno_riferimento_scheda} non trovati. Mostro i più recenti.")
                else:
                    st.warning("Dati anagrafici non disponibili."); continue
                cols = st.columns(3)
                cols[0].markdown(f"**Indirizzo:**<br>{anagrafica.get('VIA','N/D')}", unsafe_allow_html=True)
                cols[1].markdown(f"**Paese:**<br>{anagrafica.get('PAESE','N/D')}", unsafe_allow_html=True)
                cols[2].markdown(f"**Contatti:**<br>Titolare: {anagrafica.get('TITOLARE','N/D')}", unsafe_allow_html=True)
                st.divider(); st.subheader("Andamento Fatturato Annuale (da Anagrafica)")
                fatt_ann = dati_cliente.groupby('ANNO')['FATTURATO'].sum().sort_index()
                fig_bar = go.Figure(data=[go.Bar(x=fatt_ann.index, y=fatt_ann.values,
                                                 text=[format_euro_robust(v) for v in fatt_ann.values],
                                                 textposition='auto')])
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab_ordini:
        st.subheader(f"Statistiche Ordini (Anni selezionati: {', '.join(anni_scheda_selezionati)})")
        if df_ordini.empty or not anni_scheda_selezionati:
            st.info("Seleziona uno o più anni nel selettore qui sopra.")
        else:
            ordini_sel = df_ordini[(df_ordini['ANNO'].isin(anni_scheda_selezionati)) & (df_ordini['nome_cliente'].isin(clienti_selezionati))]
            if ordini_sel.empty:
                st.info("Nessun ordine trovato per i clienti e gli anni selezionati.")
            elif analysis_mode=="Aggrega Anni":
                st.subheader("Statistiche Aggregate (Clienti Selezionati)")
                total_kg = ordini_sel['KG'].sum(); total_fatt = ordini_sel['FATTURATO_ORDINE'].sum()
                pm = (total_fatt/total_kg) if total_kg>0 else 0
                a,b,c,d = st.columns(4)
                a.metric("Kg Totali", f"{total_kg:,.2f} Kg".replace(",", "."))
                b.metric("Fatturato Ordini", format_euro_robust(total_fatt))
                c.metric("Prezzo Medio Kg", f"{format_euro_robust(pm)} /Kg")
                d.metric("N. Righe Ordine", f"{len(ordini_sel)}")
                st.divider(); st.subheader("Dettaglio per Cliente")
                for cliente in clienti_selezionati:
                    with st.expander(f"Ordini per {cliente.upper()}"):
                        ord_cli = ordini_sel[ordini_sel['nome_cliente']==cliente]
                        if ord_cli.empty: st.write("Nessun dato."); continue
                        kg_cli = ord_cli['KG'].sum(); fatt_cli = ord_cli['FATTURATO_ORDINE'].sum()
                        pm_cli = (fatt_cli/kg_cli) if kg_cli>0 else 0
                        k1,k2,k3,k4 = st.columns(4)
                        k1.metric("Kg Totali", f"{kg_cli:,.2f} Kg".replace(",", "."))
                        k2.metric("Fatturato Ordini", format_euro_robust(fatt_cli))
                        k3.metric("Prezzo Medio Kg", f"{format_euro_robust(pm_cli)} /Kg")
                        k4.metric("N. Righe Ordine", f"{len(ord_cli)}")
                        c1,c2,c3 = st.columns(3)
                        top_art = ord_cli.groupby('ARTICOLO', dropna=False)['KG'].sum().sort_values(ascending=False).head(10).reset_index()
                        c1.dataframe(top_art.rename(columns={'KG':'KG Totali'}), use_container_width=True, hide_index=True)
                        top_col = ord_cli.groupby('COLORE', dropna=False)['KG'].sum().sort_values(ascending=False).head(10).reset_index()
                        c2.dataframe(top_col.rename(columns={'KG':'KG Totali'}), use_container_width=True, hide_index=True)
                        top_combo = ord_cli.groupby(['ARTICOLO','COLORE'], dropna=False)['KG'].sum().sort_values(ascending=False).head(10).reset_index()
                        c3.dataframe(top_combo.rename(columns={'KG':'KG Totali'}), use_container_width=True, hide_index=True)
                        st.markdown("##### Prezzo medio €/Kg per Articolo (aggregato)")
                        by_art = ord_cli.groupby('ARTICOLO', dropna=False).agg(KG=('KG','sum'), Fatturato=('FATTURATO_ORDINE','sum')).reset_index()
                        by_art['€/Kg'] = np.where(by_art['KG']>0, by_art['Fatturato']/by_art['KG'], 0)
                        by_art = by_art.sort_values('€/Kg', ascending=False).head(15)
                        disp = by_art[['ARTICOLO','KG','Fatturato','€/Kg']].copy()
                        disp['KG'] = disp['KG'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#","."))
                        disp['Fatturato'] = disp['Fatturato'].apply(format_euro_robust)
                        disp['€/Kg'] = disp['€/Kg'].apply(lambda v: format_euro_robust(v).replace("€ ","€ "))
                        st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("Modalità Confronto Anni: tabelle disaggregate per anno.")
                for cliente in clienti_selezionati:
                    with st.expander(f"Confronto ordini per {cliente.upper()}"):
                        oc = ordini_sel[ordini_sel['nome_cliente']==cliente]
                        if oc.empty: st.write("Nessun dato."); continue
                        st.markdown("##### Confronto Annuale per Articolo")
                        p1 = oc.pivot_table(index='ARTICOLO', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
                        p1['Totale'] = p1.sum(axis=1)
                        st.dataframe(p1.sort_values('Totale', ascending=False).drop(columns='Totale').style.format("{:,.2f}"), use_container_width=True)
                        st.markdown("##### Confronto Annuale per Colore")
                        p2 = oc.pivot_table(index='COLORE', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
                        p2['Totale'] = p2.sum(axis=1)
                        st.dataframe(p2.sort_values('Totale', ascending=False).drop(columns='Totale').style.format("{:,.2f}"), use_container_width=True)

def page_stato_dati(df_clienti, df_ordini):
    st.title("Stato dei Dati e Diagnostica")
    st.header("1. Controllo File")
    files = [p.name for p in DATA_DIR.glob('*')]
    if files: st.dataframe(files, use_container_width=True)
    else: st.error("Nessun file in 'data/'.")
    st.header("2. Analisi del Caricamento")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Anagrafica Clienti")
        if not df_clienti.empty:
            st.metric("Righe (post esplosione anni)", len(df_clienti))
            st.metric("Clienti unici", df_clienti['CLIENTE'].nunique())
        else: st.warning("File anagrafica non caricato o vuoto.")
    with c2:
        st.subheader("File Ordini")
        if not df_ordini.empty:
            st.metric("Righe totali", len(df_ordini))
            st.metric("Clienti (ordini)", df_ordini['nome_cliente'].nunique())
        else: st.warning("Nessun file ordini caricato.")
    st.header("3. Diagnosi corrispondenze")
    if not df_clienti.empty and not df_ordini.empty:
        cli_a = set(df_clienti['CLIENTE'].unique())
        cli_o = set(df_ordini['nome_cliente'].unique())
        corrisp = cli_a.intersection(cli_o); orfani = cli_o - cli_a
        st.metric("Clienti corrispondenti", len(corrisp))
        if orfani:
            st.error(f"{len(orfani)} clienti 'orfani' (ordini senza anagrafica).")
            st.dataframe(sorted([c.upper() for c in orfani]), use_container_width=True)
        else:
            st.success("Tutti i clienti degli ordini hanno una corrispondenza.")
    else:
        st.info("Carica sia anagrafica che ordini.")

# ------------------ BOOTSTRAP APP ------------------
init_db()
df_clienti = load_clients_df()
df_ordini = load_all_orders_df()
if df_clienti.empty and df_ordini.empty:
    st.error("Nessun file dati ('elenco clienti.csv' o 'ordini_*.csv') trovato nella cartella 'data'.")
    st.stop()

anni_clienti = df_clienti['ANNO'].unique() if not df_clienti.empty else []
anni_ordini = df_ordini['ANNO'].unique() if not df_ordini.empty else []
tutti_gli_anni = pd.concat([pd.Series(anni_clienti), pd.Series(anni_ordini)]).unique()
anni_disponibili = sorted([a for a in tutti_gli_anni if pd.notna(a)], reverse=True)

with st.sidebar:
    logo_path = DATA_DIR / "Logo_nyfil.png"
    if logo_path.exists(): st.image(str(logo_path), width=120)
    st.title("Navigazione")
    pagina = st.radio("Scegli una pagina:", ("Dashboard","Elenco Clienti","Analisi Dettagliata","Report Avanzati","Copilot IA","Stato dei Dati"))
    st.divider()
    st.header("Filtri Globali (non usati dal Copilot)")
    anni_selezionati_globali = st.multiselect("Anni", options=anni_disponibili, default=anni_disponibili)
    paese_selezionato = st.selectbox("Paese", options=["Tutti","Italia","Estero"])
    analysis_mode = st.radio("Modalità di Analisi Annuale", ["Aggrega Anni","Confronta Anni"], key='analysis_mode_selector')

# Routing
if pagina == "Dashboard":
    page_dashboard(df_clienti, df_ordini, anni_selezionati_globali, paese_selezionato, analysis_mode)
elif pagina == "Elenco Clienti":
    page_elenco_clienti(df_clienti, df_ordini, anni_selezionati_globali, paese_selezionato, analysis_mode)
elif pagina == "Analisi Dettagliata":
    page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode)
elif pagina == "Report Avanzati":
    page_report_avanzati(df_ordini)
elif pagina == "Copilot IA":
    # usa le funzioni Copilot avanzate che ti ho già passato
    from types import SimpleNamespace
    st.info("Apri la pagina Copilot IA dal file con le funzioni avanzate già integrate.")
elif pagina == "Stato dei Dati":
    page_stato_dati(df_clienti, df_ordini)
