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

# Province e regioni (espandibile)
PROVINCE_ITALIANE = [
    "AG","AL","AN","AO","AR","AP","AT","AV","BA","BT","BL","BN","BG","BI","BO","BZ","BS","CA","CL","CB","CE","CH","CO","CS","CR","KR","CN","EN","FE","FI","FG","FC","FR","GE","GO","GR","IM","IS","SP","AQ","LT","LE","LC","LI","LO","LU","MC","MN","MS","MT","ME","MI","MO","MB","NA","NO","NU","OR","PD","PA","PR","PV","PG","PU","PE","PC","PI","PT","PN","PZ","PO","RG","RA","RC","RE","RI","RN","RM","RO","SA","SS","SV","SI","SO","SR","TA","TE","TR","TO","TP","TN","TV","TS","UD","VA","VE","VB","VC","VR","VV","VI","VT"
]
REGIONE_TO_PROV = {
    "veneto": {"VR","VI","VE","PD","TV","BL","RO"},
    "lombardia": {"MI","MB","BG","BS","CO","CR","LC","LO","MN","PV","SO","VA","BZ","BR"},  # BZ non è Lombardia; lo lascio fuori in realtà
    "piemonte": {"AL","AT","BI","CN","NO","TO","VB","VC"},
    "emilia-romagna": {"BO","FE","FC","MO","PR","PC","RA","RE","RN"},
    "toscana": {"AR","FI","GR","LI","LU","MS","PI","PO","PT","SI"},
    "lazio": {"FR","LT","RI","RM","VT"},
}
# correzione Lombardia: rimuovo "BZ" e "BR"
REGIONE_TO_PROV["lombardia"] = {"MI","MB","BG","BS","CO","CR","LC","LO","MN","PV","SO","VA"}

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

# -------- Digest builder per IA (riuso) --------
def build_global_digest(df_clienti: pd.DataFrame, df_ordini: pd.DataFrame,
                        anni_sel: list, paese_sel: str, top_n: int = 10) -> str:
    anni_sel = [str(a) for a in anni_sel] if anni_sel else []
    dfc = df_clienti[df_clienti['ANNO'].isin(anni_sel)] if anni_sel else df_clienti.copy()
    if paese_sel != "Tutti":
        dfc = dfc[dfc['PAESE']==paese_sel]
    total_revenue = float(dfc['FATTURATO'].sum()) if not dfc.empty else 0.0
    quota_italia = float(dfc[dfc['PAESE']=='Italia']['FATTURATO'].sum()/total_revenue*100) if total_revenue>0 else 0.0
    n_clienti = int(dfc['CLIENTE'].nunique()) if not dfc.empty else 0
    top_clients = (dfc.groupby('CLIENTE')['FATTURATO'].sum().sort_values(ascending=False).head(top_n))
    dfo = df_ordini[df_ordini['ANNO'].isin(anni_sel)] if anni_sel else df_ordini.copy()
    kg_by_client = dfo.groupby('nome_cliente')['KG'].sum()
    top_art = (dfo.groupby('ARTICOLO')['KG'].sum().sort_values(ascending=False).head(10))
    lines = []
    lines.append(f"FILTRI → Anni: {', '.join(anni_sel) or 'tutti'} | Paese: {paese_sel}")
    lines.append(f"KPI → Fatturato Totale: {format_euro_robust(total_revenue)} | Quota Italia: {quota_italia:.1f}% | N° Clienti: {n_clienti}")
    lines.append("Top Clienti (Fatturato / Kg):")
    for cli, fatt in top_clients.items():
        kg = float(kg_by_client.get(cli, 0))
        lines.append(f" - {cli.upper()}: {format_euro_robust(float(fatt))} | {kg:,.2f} Kg".replace(",", "."))
    lines.append(f"Top Articoli per Kg: {dict(top_art)}")
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

# ------------------ PARSER NATURALE (Copilot) ------------------
def parse_user_query(q: str):
    """Estrae intent basilari dalla domanda: anni, top_n, metrica, area."""
    qlow = q.lower()

    # anni (4 cifre 20xx)
    anni = re.findall(r'\b(20\d{2})\b', qlow)
    anni = list(dict.fromkeys(anni))  # unique, order-preserving

    # top N
    top_n = None
    m = re.search(r'\btop\s+(\d+)\b', qlow) or re.search(r'\b(primi|migliori?)\s+(\d+)\b', qlow)
    if m:
        top_n = int(m.groups()[-1])
    elif re.search(r'\b(più|maggior|massimo|max)\b', qlow):
        top_n = 1

    # metrica
    metric = None
    if "profittevol" in qlow or "redditizi" in qlow:
        metric = "euro_kg"
    elif "€/kg" in qlow or "euro/kg" in qlow or "prezzo medio" in qlow:
        metric = "euro_kg"
    elif "fatturato" in qlow:
        metric = "fatturato"
    elif re.search(r'\bkg\b|\bquantit', qlow):
        metric = "kg"

    # area
    paese = None
    if "italia" in qlow: paese = "Italia"
    elif "estero" in qlow: paese = "Estero"

    regione = None
    for reg in REGIONE_TO_PROV.keys():
        if reg in qlow:
            regione = reg
            break

    # entità (clienti/articoli)
    entity = "clienti"
    if "articol" in qlow: entity = "articoli"
    if "color" in qlow: entity = "colori"

    return {
        "anni": anni,              # list di stringhe
        "top_n": top_n,            # int o None
        "metric": metric,          # 'euro_kg' | 'fatturato' | 'kg' | None
        "paese": paese,            # 'Italia' | 'Estero' | None
        "regione": regione,        # key del dict REGIONE_TO_PROV o None
        "entity": entity
    }

def apply_region_filter(df_clienti: pd.DataFrame, regione_key: str) -> set:
    """Ritorna l'insieme di clienti in una certa regione (via PROVINCIA)."""
    if not regione_key or 'PROVINCIA' not in df_clienti.columns:
        return set()
    provs = REGIONE_TO_PROV.get(regione_key, set())
    if not provs: return set()
    return set(df_clienti[df_clienti['PROVINCIA'].str.upper().isin(provs)]['CLIENTE'].unique())

def compute_copilot_answer(df_clienti: pd.DataFrame, df_ordini: pd.DataFrame, intent: dict):
    """Calcola risposta deterministica in base all'intent estratto."""
    anni = intent.get("anni") or None
    metric = intent.get("metric") or "fatturato"
    entity = intent.get("entity") or "clienti"
    top_n = intent.get("top_n") or 3
    paese = intent.get("paese")
    regione = intent.get("regione")

    # Filtri base su TUTTA la base (svincolato da sidebar)
    dfc = df_clienti.copy()
    if anni:
        dfc = dfc[dfc['ANNO'].isin(anni)]
    if paese:
        dfc = dfc[dfc['PAESE']==paese]

    clients_region = None
    if regione:
        clients_region = apply_region_filter(dfc, regione)

    # --- CLIENTI ---
    if entity == "clienti":
        # costruisco aggregati per metriche
        # fatturato da anagrafica
        fatt = dfc.groupby('CLIENTE', as_index=False)['FATTURATO'].sum()

        # ordini (kg e fatt ordini)
        dfo = df_ordini.copy()
        if anni:
            dfo = dfo[dfo['ANNO'].isin(anni)]
        # restringi a clienti esistenti (coerenza nomi)
        if not dfc.empty:
            dfo = dfo[dfo['nome_cliente'].isin(set(dfc['CLIENTE']))]

        kg = dfo.groupby('nome_cliente', as_index=False)['KG'].sum()
        fatt_o = dfo.groupby('nome_cliente', as_index=False)['FATTURATO_ORDINE'].sum()

        # unione
        base = pd.merge(fatt, kg, left_on='CLIENTE', right_on='nome_cliente', how='outer')
        base = pd.merge(base, fatt_o, on='nome_cliente', how='outer')
        base['CLIENTE'] = base['CLIENTE'].fillna(base['nome_cliente'])
        base.drop(columns=['nome_cliente'], inplace=True)
        base[['FATTURATO','KG','FATTURATO_ORDINE']] = base[['FATTURATO','KG','FATTURATO_ORDINE']].fillna(0.0)

        # filtro regione se richiesto
        if clients_region is not None and len(clients_region)>0:
            base = base[base['CLIENTE'].isin(clients_region)]

        # metrica richiesta
        if metric == "fatturato":
            base['METRICA'] = base['FATTURATO']
            label_metric = "Fatturato"
            fmt = base['METRICA'].apply(format_euro_robust)
        elif metric == "kg":
            base['METRICA'] = base['KG']
            label_metric = "Kg"
            fmt = base['METRICA'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
        else:  # euro_kg (profittevole)
            base['€/Kg'] = np.where(base['KG']>0, base['FATTURATO_ORDINE']/base['KG'], np.nan)
            # soglia minima kg per robustezza
            base['METRICA'] = np.where(base['KG']>=20, base['€/Kg'], np.nan)
            base = base.dropna(subset=['METRICA'])
            label_metric = "€/Kg medio (soglia ≥ 20 Kg)"
            fmt = base['METRICA'].map(lambda v: format_euro_robust(v).replace("€ ", "€ "))

        if base.empty:
            return {"table": pd.DataFrame(), "explain":"Nessun risultato con i filtri dedotti.", "assumption": intent}

        out = base.sort_values('METRICA', ascending=False).head(top_n).copy()
        out_display = pd.DataFrame({
            "CLIENTE": out['CLIENTE'].str.upper(),
            label_metric: fmt.loc[out.index],
            "Fatturato Totale": out['FATTURATO'].apply(format_euro_robust),
            "Kg Totali": out['KG'].map(lambda x: f"{x:,.2f} Kg".replace(",", ".")),
            "Fatturato Ordini": out['FATTURATO_ORDINE'].apply(format_euro_robust)
        })
        explain = []
        explain.append(f"Metrica: **{label_metric}** | Entità: **Clienti** | Top {top_n}")
        explain.append(f"Anni: **{(', '.join(anni)) if anni else 'tutti'}**")
        if paese: explain.append(f"Paese: **{paese}**")
        if regione: explain.append(f"Regione: **{regione.title()}**")
        if metric=="euro_kg":
            explain.append("Definizione *profittevole*: €/Kg medio più alto; applicata soglia **≥ 20 Kg** per evitare outlier.")
        return {"table": out_display, "explain":" • ".join(explain), "assumption": intent}

    # --- ARTICOLI / COLORI (estendibile) ---
    # Per ora implemento logica base su articoli per kg/fatturato.
    dfo = df_ordini.copy()
    if anni:
        dfo = dfo[dfo['ANNO'].isin(anni)]
    # Paese/regione: mappo ai clienti appartenenti a quell'area
    if paese or regione:
        dfc_area = df_clienti.copy()
        if anni: dfc_area = dfc_area[dfc_area['ANNO'].isin(anni)]
        if paese: dfc_area = dfc_area[dfc_area['PAESE']==paese]
        if regione:
            clients_region = apply_region_filter(dfc_area, regione)
            if clients_region:
                dfc_area = dfc_area[dfc_area['CLIENTE'].isin(clients_region)]
        allowed_clients = set(dfc_area['CLIENTE'])
        dfo = dfo[dfo['nome_cliente'].isin(allowed_clients)]

    if entity == "articoli":
        grp_key = "ARTICOLO"
    else:
        grp_key = "COLORE"

    agg = dfo.groupby(grp_key, as_index=False).agg(KG=('KG','sum'), FATTURATO=('FATTURATO_ORDINE','sum'))
    if metric == "fatturato":
        agg['METRICA'] = agg['FATTURATO']
        label_metric = "Fatturato"
        fmt = agg['METRICA'].apply(format_euro_robust)
    elif metric == "euro_kg":
        agg['METRICA'] = np.where(agg['KG']>0, agg['FATTURATO']/agg['KG'], np.nan)
        agg = agg.dropna(subset=['METRICA'])
        label_metric = "€/Kg medio"
        fmt = agg['METRICA'].map(lambda v: format_euro_robust(v))
    else:
        agg['METRICA'] = agg['KG']
        label_metric = "Kg"
        fmt = agg['METRICA'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))

    out = agg.sort_values('METRICA', ascending=False).head(top_n).copy()
    if out.empty:
        return {"table": pd.DataFrame(), "explain":"Nessun risultato con i filtri dedotti.", "assumption": intent}

    out_display = pd.DataFrame({
        grp_key.upper(): out[grp_key],
        label_metric: fmt.loc[out.index],
        "Kg Totali": out['KG'].map(lambda x: f"{x:,.2f} Kg".replace(",", ".")),
        "Fatturato": out['FATTURATO'].apply(format_euro_robust)
    })
    explain = []
    explain.append(f"Metrica: **{label_metric}** | Entità: **{entity.title()}** | Top {top_n}")
    explain.append(f"Anni: **{(', '.join(anni)) if anni else 'tutti'}**")
    if paese: explain.append(f"Paese: **{paese}**")
    if regione: explain.append(f"Regione: **{regione.title()}**")
    return {"table": out_display, "explain":" • ".join(explain), "assumption": intent}

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

def page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode):
    st.title("Analisi Dettagliata Cliente")
    clienti_options = sorted(df_clienti['CLIENTE'].str.upper().unique())
    clienti_selezionati_upper = st.multiselect("Seleziona uno o più clienti per l'analisi", options=clienti_options, key='client_selector_detail')
    if not clienti_selezionati_upper:
        st.info("Seleziona uno o più clienti per iniziare l'analisi.")
        return
    clienti_selezionati = [c.lower() for c in clienti_selezionati_upper]
    anni_scheda_selezionati = st.multiselect("Seleziona anni per l'analisi di dettaglio", options=anni_disponibili, default=anni_selezionati_globali, key="anni_dettaglio_selector")
    st.header(f"Scheda Alleati: {', '.join(clienti_selezionati_upper)}")
    anno_rif = anni_scheda_selezionati[0] if anni_scheda_selezionati else anni_disponibili[0]
    tab_eval, tab_dati, tab_ordini = st.tabs(["Valutazione Alleati","Anagrafica & Fatturato","Ordini & Statistiche"])

    with tab_eval:
        st.subheader(f"Valutazioni Individuali (Anno di riferimento: {anno_rif})")
        evals_data = {}
        for cliente in clienti_selezionati:
            with st.expander(f"Valutazione per {cliente.upper()}"):
                with st.form(key=f"evaluation_form_{cliente}_{anno_rif}"):
                    ev = load_evaluation(cliente, anno_rif)
                    cols = st.columns(3); temp={}
                    for i,q in enumerate(EVALUATION_QUESTIONS):
                        with cols[i%3]:
                            temp[q['key']] = st.slider(q['text'],1,5,value=ev.get(q['key'],3), key=f"{q['key']}_{cliente}_{anno_rif}")
                    if st.form_submit_button("Salva Valutazione"):
                        save_evaluation(cliente, anno_rif, temp); st.cache_data.clear(); st.rerun()
                evals_data[cliente] = load_evaluation(cliente, anno_rif)

                st.markdown("---")
                if st.button("🔎 Analizza con IA", key=f"ai_analyze_{cliente}_{anno_rif}"):
                    with st.spinner("Sto analizzando i dati con l'IA..."):
                        # digest sintetico per cliente
                        def build_client_digest(cliente, anni_sel, anno_rif, dfc, dfo):
                            anni_sel = [str(a) for a in anni_sel] if anni_sel else []
                            eval_data = load_evaluation(cliente, anno_rif)
                            tot, val_ec, val_rel = calculate_scores(eval_data)
                            fatt_cli = (dfc[(dfc['CLIENTE']==cliente)&(dfc['ANNO'].isin(anni_sel))]
                                        .groupby('ANNO',as_index=False)['FATTURATO'].sum().sort_values('ANNO'))
                            ord_cli = dfo[(dfo['nome_cliente']==cliente) & (dfo['ANNO'].isin(anni_sel))]
                            kg_tot = float(ord_cli['KG'].sum()); fatt_o = float(ord_cli['FATTURATO_ORDINE'].sum())
                            prezzo = (fatt_o/kg_tot) if kg_tot>0 else 0
                            lines=[]
                            lines.append(f"Cliente: {cliente.upper()} | Anno valutazione: {anno_rif} | Anni: {', '.join(anni_sel) or 'tutti'}")
                            lines.append(f"Valutazione → Tot: {tot:.1f} | Econ: {val_ec:.2f} | Rel: {val_rel:.2f}")
                            lines.append(f"Fatturato per anno: { {r['ANNO']: r['FATTURATO'] for _,r in fatt_cli.iterrows()} }")
                            lines.append(f"Ordini → Kg: {kg_tot:.2f} | Fatt: {fatt_o:.2f} | €/Kg: {prezzo:.3f}")
                            return "\n".join(lines)
                        digest = build_client_digest(cliente, anni_scheda_selezionati, anno_rif, df_clienti, df_ordini)
                        system = ("Sei un business analyst per un'azienda B2B (nylon). Usa SOLO i dati forniti. "
                                  "Output in italiano, conciso, a punti: 1) Sintesi 2) Opportunità 3) Rischi 4) Azioni 30/60/90 gg.")
                        user = f"DATI CLIENTE (digest):\n{digest}"
                        out = call_llm(system, user)
                        st.markdown("#### Risultato IA"); st.markdown(out)

        st.divider(); st.subheader("Analisi Strategica Comparata")
        fig_matrix = go.Figure(); fig_radar = go.Figure()
        for cliente,data in evals_data.items():
            _, ve, vr = calculate_scores(data)
            fig_matrix.add_trace(go.Scatter(x=[ve], y=[vr], mode='markers+text', text=cliente.upper(), marker=dict(size=15), name=cliente.upper()))
            radar_values = [data[q['key']] for q in EVALUATION_QUESTIONS]
            fig_radar.add_trace(go.Scatterpolar(r=radar_values+[radar_values[0]],
                                                theta=[f"Q{i+1}" for i in range(15)]+["Q1"],
                                                fill='toself', name=cliente.upper(), opacity=0.7))
        c1,c2 = st.columns(2)
        with c1: st.markdown("##### Matrice Decisionale"); st.plotly_chart(fig_matrix, use_container_width=True)
        with c2: st.markdown("##### Profili Radar"); st.plotly_chart(fig_radar, use_container_width=True)

    with tab_dati:
        for cliente in clienti_selezionati:
            with st.expander(f"Dati per {cliente.upper()}"):
                dati_cliente = df_clienti[df_clienti['CLIENTE']==cliente]
                st.subheader(f"Anagrafica (Riferimento anno: {anno_rif})")
                anag_anno = dati_cliente[dati_cliente['ANNO']==anno_rif]
                if not anag_anno.empty: anagrafica = anag_anno.iloc[0]
                elif not dati_cliente.empty:
                    anagrafica = dati_cliente.sort_values('ANNO', ascending=False).iloc[0]
                    st.info(f"Dati anagrafici per l'anno {anno_rif} non trovati. Mostro i più recenti.")
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

# --- REPORT AVANZATI (Predittivo articolo) ---
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

# --- NUOVA PAGINA: COPILOT IA (svincolato dai filtri) ---
def page_copilot(df_clienti: pd.DataFrame, df_ordini: pd.DataFrame, anni_disponibili: list):
    st.title("🤖 Copilot IA — Query Libera su Tutta la Dashboard")
    st.caption("Il Copilot interpreta la domanda, imposta i filtri necessari (anno, metrica, area) e calcola la risposta sui dati completi, indipendentemente dai filtri globali.")
    q = st.text_input("Scrivi la tua domanda (es. 'Qual è stato il cliente più profittevole in Veneto nel 2025?')",
                      key="copilot_free_q")
    colx, coly = st.columns([1,1])
    with colx:
        add_ai_comment = st.checkbox("Aggiungi commento IA (opzionale)", value=False)
    with coly:
        threshold = st.number_input("Soglia minima Kg per 'profittevole' (€/Kg)", value=20.0, min_value=0.0, step=5.0)

    if st.button("Esegui"):
        if not q.strip():
            st.warning("Inserisci una domanda."); return
        # 1) Parsing rapido
        intent = parse_user_query(q)
        # 2) Calcolo deterministico
        res = compute_copilot_answer(df_clienti, df_ordini, intent)
        st.markdown(f"**Interpretazione automatica:** `{json.dumps(intent, ensure_ascii=False)}`")
        if res["table"].empty:
            st.warning("Nessun risultato con i filtri dedotti.")
        else:
            st.markdown(res["explain"])
            st.dataframe(res["table"], use_container_width=True, hide_index=True)

        # 3) (Opzionale) Commento IA sulla risposta
        if add_ai_comment:
            anni_txt = ", ".join(intent.get("anni") or []) or "tutti"
            paese = intent.get("paese") or "Tutti"
            regione = intent.get("regione") or "-"
            metric = intent.get("metric") or "fatturato"
            entity = intent.get("entity") or "clienti"
            sample_table = res["table"].head(5).to_dict(orient="records") if not res["table"].empty else []
            digest = [
                f"Anni: {anni_txt}", f"Paese: {paese}", f"Regione: {regione}",
                f"Entità: {entity}", f"Metrica: {metric}", f"Tabella (top): {sample_table}",
                f"Soglia profittevole (kg): {threshold}"
            ]
            system = ("Sei un assistente analitico per una dashboard B2B (nylon). "
                      "Commenta brevemente il risultato mostrato (max 6 bullet), in italiano, "
                      "spiegando eventuali limiti/assunzioni e suggerendo un'azione.")
            user = " | ".join(digest) + f"\nDomanda utente: {q}"
            ai_out = call_llm(system, user)
            st.markdown("**Commento IA:**")
            st.markdown(ai_out)

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
    page_copilot(df_clienti, df_ordini, anni_disponibili)
elif pagina == "Stato dei Dati":
    page_stato_dati(df_clienti, df_ordini)
