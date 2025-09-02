# app.py
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

# ===================== CONFIG & PATHS =====================
st.set_page_config(page_title="Analisi Clienti Nyfil", layout="wide")

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"          # CSV/immagini (repo)
DATA_DIR.mkdir(exist_ok=True)

APP_DATA_DIR = Path("/mount/data/nyfil")  # ✅ scrivibile su Streamlit Cloud
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = APP_DATA_DIR / "app.db"     # ✅ DB persistente
CLIENTS_CSV = DATA_DIR / "elenco clienti.csv"

PROVINCE_ITALIANE = [
    "AG","AL","AN","AO","AR","AP","AT","AV","BA","BT","BL","BN","BG","BI","BO","BZ","BS","CA","CL","CB","CE","CH","CO","CS","CR","KR","CN","EN",
    "FE","FI","FG","FC","FR","GE","GO","GR","IM","IS","SP","AQ","LT","LE","LC","LI","LO","LU","MC","MN","MS","MT","ME","MI","MO","MB","NA","NO",
    "NU","OR","PD","PA","PR","PV","PG","PU","PE","PC","PI","PT","PN","PZ","PO","RG","RA","RC","RE","RI","RN","RM","RO","SA","SS","SV","SI","SO",
    "SR","TA","TE","TR","TO","TP","TN","TV","TS","UD","VA","VE","VB","VC","VR","VV","VI","VT"
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

# ===================== UTILS =====================
def format_euro_robust(v):
    try:
        if pd.isna(v) or not isinstance(v,(int,float)): return "N/A"
        return f"€ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")
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

# ===================== DB INIT & FALLBACK =====================
def init_db():
    conn = get_db_connection()
    try:
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
                    digest TEXT, output TEXT, created_at TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_actions (
                    cliente TEXT NOT NULL, anno_rif TEXT NOT NULL,
                    stars INTEGER DEFAULT 0, priority REAL DEFAULT 0,
                    note TEXT, done INTEGER DEFAULT 0, updated_at TEXT,
                    PRIMARY KEY (cliente, anno_rif)
                );
            """)
        st.session_state["_DB_FALLBACK"] = False
    except sqlite3.OperationalError:
        st.warning("DB non scrivibile: uso memoria volatile (si perderà al riavvio).")
        st.session_state["_DB_FALLBACK"] = True
        st.session_state.setdefault("_MEM_EVAL", {})
        st.session_state.setdefault("_MEM_AI", [])
        st.session_state.setdefault("_MEM_ACT", {})

def _db_fallback() -> bool:
    return st.session_state.get("_DB_FALLBACK", False)

# ===================== LOAD DATA =====================
@st.cache_data
def load_clients_df(uploaded_file=None) -> pd.DataFrame:
    source = CLIENTS_CSV if CLIENTS_CSV.exists() else uploaded_file
    if not source: return pd.DataFrame()
    try:
        df = pd.read_csv(source, sep=';', encoding='latin1', low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        mapping = {
            "nome_clietne":"CLIENTE","nome_cliente":"CLIENTE","via":"VIA","città":"CITTA","cap":"CAP",
            "provincia":"PROVINCIA","titolare_azienda":"TITOLARE","recapiti_mail":"EMAIL",
            "anno":"ANNO_ORIG","imponibile":"IMPONIBILE"
        }
        df.rename(columns=mapping, inplace=True)
        if 'CLIENTE' in df.columns:
            df['CLIENTE'] = df['CLIENTE'].apply(clean_customer_name)
        df['ANNO_ORIG'] = df['ANNO_ORIG'].astype(str).str.split('/')
        df = df.explode('ANNO_ORIG')
        df['ANNO'] = df['ANNO_ORIG'].apply(normalize_year)
        df['FATTURATO'] = df['IMPONIBILE'].apply(parse_decimal_string)
        df['PAESE'] = df.apply(lambda r: detect_country(r.get('CAP',''), r.get('PROVINCIA','')), axis=1)
        agg = {k:'first' for k in ['VIA','CITTA','CAP','PROVINCIA','TITOLARE','EMAIL','PAESE']}
        agg['FATTURATO'] = 'sum'
        agg = {k:v for k,v in agg.items() if k in df.columns}
        return df.groupby(['CLIENTE','ANNO']).agg(agg).reset_index()
    except Exception as e:
        st.error(f"Errore lettura file clienti: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all_orders_df() -> pd.DataFrame:
    files = glob.glob(str(DATA_DIR / "ordini_*"))
    if not files: return pd.DataFrame()
    dfs=[]
    for f in files:
        try:
            y = re.search(r'(\d+)', Path(f).stem)
            if not y: continue
            year = normalize_year(y.group(1))
            df = pd.read_csv(f, sep=';', encoding='latin1', low_memory=False) if f.endswith('.csv') else pd.read_excel(f)
            df.columns = [re.sub(r'_\d+$','',c).strip().lower() for c in df.columns]
            df['ANNO'] = year
            dfs.append(df)
        except Exception as e:
            st.warning(f"Impossibile leggere {f}: {e}")
    if not dfs: return pd.DataFrame()
    dfo = pd.concat(dfs, ignore_index=True)
    if 'nome_cliente' in dfo.columns:
        dfo['nome_cliente'] = dfo['nome_cliente'].apply(clean_customer_name)
    dfo.dropna(subset=['articolo_colore','quantita'], inplace=True)
    dfo = dfo[dfo['quantita'] != 0].copy()
    dfo['FATTURATO_ORDINE'] = dfo['imponibile'].apply(parse_decimal_string) if 'imponibile' in dfo.columns else 0
    dfo['KG'] = dfo['quantita'].apply(parse_decimal_string).round(2)
    spl = dfo['articolo_colore'].str.rsplit(' - ', n=1, expand=True)
    dfo['ARTICOLO'] = spl[0].str.strip()
    dfo['COLORE'] = spl[1].str.strip().fillna('NON SPECIFICATO')
    return dfo[['nome_cliente','ANNO','ARTICOLO','COLORE','KG','FATTURATO_ORDINE']]

# ===================== EVALUATION & ACTIONS =====================
def load_evaluation(cliente: str, anno: str) -> dict:
    if _db_fallback():
        return st.session_state["_MEM_EVAL"].get((cliente, anno), {q['key']:3 for q in EVALUATION_QUESTIONS})
    conn = get_db_connection()
    cur = conn.cursor()
    keys = [q['key'] for q in EVALUATION_QUESTIONS]
    cur.execute(f"SELECT {', '.join(keys)} FROM evaluation WHERE cliente=? AND anno=?", (cliente, anno))
    row = cur.fetchone()
    if row: return dict(zip(keys, row))
    return {k:3 for k in keys}

def save_evaluation(cliente: str, anno: str, data: dict):
    if _db_fallback():
        st.session_state["_MEM_EVAL"][(cliente, anno)] = data.copy()
        st.toast("Valutazione salvata (in memoria).")
        return
    conn = get_db_connection()
    keys = [q['key'] for q in EVALUATION_QUESTIONS]
    cols = ", ".join(keys)
    placeholders = ", ".join(["?"]*len(keys))
    q = f"INSERT OR REPLACE INTO evaluation (cliente, anno, {cols}, updated_at) VALUES (?, ?, {placeholders}, ?)"
    vals = [cliente, anno] + [data.get(k,1) for k in keys] + [datetime.now().isoformat()]
    with conn:
        conn.execute(q, tuple(vals))
    st.toast(f"Valutazione per {cliente.upper()} ({anno}) salvata!")

def calculate_scores(ev):
    val_econ = [q['key'] for q in EVALUATION_QUESTIONS if q['category']=="Valore Economico"]
    val_rel  = [q['key'] for q in EVALUATION_QUESTIONS if q['category']=="Valore Relazionale"]
    total = sum(ev.values())
    ve = float(np.mean([ev[k] for k in val_econ]))
    vr = float(np.mean([ev[k] for k in val_rel]))
    return total, ve, vr

def get_matrix_quadrant(x,y):
    if x>3 and y>3: return "Partner Chiave"
    if x>3 and y<=3: return "Specialista Redditizio"
    if x<=3 and y>3: return "Amico a Basso Impatto"
    return "Cliente Marginale"

def stars_from_total(t):
    return 5 if t>=65 else 4 if t>=55 else 3 if t>=45 else 2 if t>=35 else 1

def priority_from_quadrant(ve, vr):
    quad = get_matrix_quadrant(ve, vr)
    if quad == "Specialista Redditizio": base = 5.0
    elif quad == "Amico a Basso Impatto": base = 4.0
    elif quad == "Partner Chiave": base = 2.0
    else: base = 3.0
    adj = (5 - vr)*0.5 + (5 - min(ve,5))*0.3
    return round(base + adj, 2)

def upsert_client_action(cliente, anno_rif, stars=None, priority=None, note=None, done=None):
    now = datetime.now().isoformat()
    if _db_fallback():
        row = st.session_state["_MEM_ACT"].get((cliente, anno_rif), {"stars":0,"priority":0.0,"note":"","done":0,"updated_at":now})
        if stars is not None: row["stars"]=int(stars)
        if priority is not None: row["priority"]=float(priority)
        if note is not None: row["note"]=note
        if done is not None: row["done"]=int(done)
        row["updated_at"]=now
        st.session_state["_MEM_ACT"][(cliente, anno_rif)] = row
        return
    conn = get_db_connection()
    with conn:
        ex = conn.execute("SELECT 1 FROM client_actions WHERE cliente=? AND anno_rif=?", (cliente, anno_rif)).fetchone()
        if not ex:
            conn.execute("INSERT INTO client_actions (cliente, anno_rif, stars, priority, note, done, updated_at) VALUES (?,?,?,?,?,?,?)",
                         (cliente, anno_rif, stars or 0, priority or 0.0, note or "", done or 0, now))
        else:
            sets=[]; vals=[]
            if stars is not None: sets.append("stars=?"); vals.append(int(stars))
            if priority is not None: sets.append("priority=?"); vals.append(float(priority))
            if note is not None: sets.append("note=?"); vals.append(note)
            if done is not None: sets.append("done=?"); vals.append(int(done))
            sets.append("updated_at=?"); vals.append(now)
            sql = f"UPDATE client_actions SET {', '.join(sets)} WHERE cliente=? AND anno_rif=?"
            vals += [cliente, anno_rif]
            conn.execute(sql, tuple(vals))

def load_actions_table(anno_rif):
    if _db_fallback():
        rows=[]
        for (cli, ar), r in st.session_state["_MEM_ACT"].items():
            if ar == anno_rif: rows.append({"cliente":cli,"anno_rif":ar, **r})
        return pd.DataFrame(rows)
    conn = get_db_connection()
    return pd.read_sql_query("SELECT * FROM client_actions WHERE anno_rif = ?", conn, params=[anno_rif])

# ===================== LLM =====================
@st.cache_resource
def get_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        return None
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MODEL_NAME = st.secrets.get("MODEL_NAME","gpt-4o-mini")
TEMPERATURE = float(st.secrets.get("MODEL_TEMPERATURE",0.2))
ANON_SALT = st.secrets.get("ANON_SALT","nyfil")

def _truncate(s, max_chars=15000):
    s = str(s)
    return s if len(s)<=max_chars else s[:max_chars] + "\n... [troncato]"

@retry(wait=wait_random_exponential(min=1, max=6), stop=stop_after_attempt(3))
def call_llm(system_prompt, user_prompt):
    client = get_openai_client()
    if client is None:
        return "⚠️ Configura OPENAI_API_KEY nei secrets della app."
    r = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":_truncate(user_prompt)}]
    )
    return r.choices[0].message.content.strip()

def anon_client_id(name):
    h = hashlib.sha256((ANON_SALT + (name or "")).encode("utf-8")).hexdigest()[:8].upper()
    return f"CLIENTE_{h}"

def save_ai_output(cliente, anno_rif, digest, output):
    rec = {"cliente":cliente,"anno_rif":anno_rif,"digest":digest,"output":output,"created_at":datetime.now().isoformat()}
    if _db_fallback():
        st.session_state["_MEM_AI"].append(rec); return
    conn = get_db_connection()
    with conn:
        conn.execute("INSERT INTO ai_insights (cliente, anno_rif, digest, output, created_at) VALUES (?,?,?,?,?)",
                     (cliente, anno_rif, digest, output, rec["created_at"]))

def load_last_ai_output(cliente, anno_rif):
    if _db_fallback():
        items = [r for r in st.session_state["_MEM_AI"] if r["cliente"]==cliente and r["anno_rif"]==anno_rif]
        return {"output":items[-1]["output"], "created_at":items[-1]["created_at"]} if items else None
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT output, created_at FROM ai_insights WHERE cliente=? AND anno_rif=? ORDER BY id DESC LIMIT 1", (cliente, anno_rif))
    row = cur.fetchone()
    return {"output":row[0], "created_at":row[1]} if row else None

def build_client_digest_anon(cliente, anni_sel, anno_rif, dfc, dfo):
    alias = anon_client_id(cliente)
    anni_sel = [str(a) for a in anni_sel] if anni_sel else []
    ev = load_evaluation(cliente, anno_rif)
    tot, ve, vr = calculate_scores(ev)
    fatt = (dfc[(dfc['CLIENTE']==cliente) & (dfc['ANNO'].isin(anni_sel))]
            .groupby('ANNO', as_index=False)['FATTURATO'].sum().sort_values('ANNO'))
    ords = dfo[(dfo['nome_cliente']==cliente) & (dfo['ANNO'].isin(anni_sel))]
    kg = float(ords['KG'].sum()); fatt_o = float(ords['FATTURATO_ORDINE'].sum())
    pmedio = (fatt_o/kg) if kg>0 else 0
    lines = [
        f"Cliente (anonimo): {alias} | Anno valutazione: {anno_rif} | Anni: {', '.join(anni_sel) or 'tutti'}",
        f"Valutazione → Tot:{tot:.1f} | Econ:{ve:.2f} | Rel:{vr:.2f}",
        f"Fatturato per anno (€): { {r['ANNO']: float(r['FATTURATO']) for _,r in fatt.iterrows()} }",
        f"Ordini → Kg:{kg:.2f} | Fatt:{fatt_o:.2f} | €/Kg:{pmedio:.3f}"
    ]
    return "\n".join(lines)

# ===================== NUOVI CLIENTI =====================
@st.cache_data
def compute_first_year_per_client(df_clienti, df_ordini):
    first_a = df_clienti.groupby('CLIENTE')['ANNO'].min().reset_index().rename(columns={'ANNO':'FIRST_YEAR'})
    if not df_ordini.empty:
        first_o = df_ordini.groupby('nome_cliente')['ANNO'].min().reset_index().rename(columns={'nome_cliente':'CLIENTE','ANNO':'FIRST_YEAR_ORD'})
        first = pd.merge(first_a, first_o, on='CLIENTE', how='outer')
        first['FIRST_YEAR'] = first[['FIRST_YEAR','FIRST_YEAR_ORD']].min(axis=1)
        first.drop(columns=['FIRST_YEAR_ORD'], inplace=True)
    else:
        first = first_a
    return first

# ===================== COPILOT ENGINE =====================
def parse_user_query(q: str):
    txt = q.lower()
    intent = 'ranking'
    if re.search(r'\bcresc|aument|trend|variaz|cagr|yoy\b', txt): intent='growth'
    elif re.search(r'\bquota|percentual|%\b', txt): intent='share'
    elif re.search(r'\bmediana\b', txt): intent='median'
    elif re.search(r'\bmedio|media|average\b', txt): intent='mean'
    metric=None
    if re.search(r'€/kg|euro/kg|profittevol', txt): metric='euro_kg'
    elif re.search(r'\bkg\b|\bquantit', txt): metric='kg'
    elif re.search(r'fatturat', txt): metric='fatturato'
    entity='clienti'
    if 'articol' in txt: entity='articoli'
    elif 'color' in txt: entity='colori'
    years = re.findall(r'\b(20\d{2})\b', txt)
    anni = list(dict.fromkeys(years))
    mrange = re.search(r'(dal|da|tra|fra)\s*(20\d{2})\s*(al|a|e)\s*(20\d{2})', txt)
    anni_range=None
    if mrange:
        a1,a2 = mrange.group(2), mrange.group(4)
        if a1<=a2: anni_range={'from':a1,'to':a2}
    if re.search(r'tutt[ioa] gli anni|tutti i periodi', txt): anni, anni_range = [], None
    top_n=None
    mt = re.search(r'\btop\s+(\d+)\b', txt) or re.search(r'\b(primi|migliori?)\s+(\d+)\b', txt)
    if mt: top_n=int(mt.groups()[-1])
    elif re.search(r'\bpiù|massim|maggior|rank 1\b', txt): top_n=1
    paese = 'Italia' if 'italia' in txt else ('Estero' if 'estero' in txt else None)
    regione=None
    for reg in REGIONE_TO_PROV.keys():
        if reg in txt: regione=reg; break
    thresholds=[]
    for pat in [r'(>=|<=|>|<)\s*€?\s*([\d\.,]+)\s*(€/kg|euro/kg|kg)?',
                r'(sopra|oltre|maggiore di)\s*€?\s*([\d\.,]+)\s*(€/kg|euro/kg|kg)?',
                r'(sotto|inferiore a|minore di)\s*€?\s*([\d\.,]+)\s*(€/kg|euro/kg|kg)?']:
        for m in re.finditer(pat, txt):
            op_raw=m.group(1); val=float(m.group(2).replace('.','').replace(',','.')); unit=m.group(3) or ''
            if op_raw in ('sopra','oltre','maggiore di'): op='>'
            elif op_raw in ('sotto','inferiore a','minore di'): op='<'
            else: op=op_raw
            field='fatturato'
            if 'kg' in unit: field='kg'
            if '€/kg' in unit or 'euro/kg' in unit: field='euro_kg'
            thresholds.append({'field':field,'op':op,'value':val})
    return {'intent':intent,'metric':metric,'entity':entity,'anni':anni,'anni_range':anni_range,'top_n':top_n,'paese':paese,'regione':regione,'thresholds':thresholds}

def _allowed_clients_by_area(df_clienti, anni_list=None, anni_range=None, paese=None, regione=None):
    dfc = df_clienti.copy()
    if anni_list: dfc = dfc[dfc['ANNO'].isin(anni_list)]
    if anni_range: dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
    if paese: dfc = dfc[dfc['PAESE']==paese]
    if regione and 'PROVINCIA' in dfc.columns:
        provs = REGIONE_TO_PROV.get(regione, set())
        if provs: dfc = dfc[dfc['PROVINCIA'].str.upper().isin(provs)]
    return set(dfc['CLIENTE'])

def _apply_thresholds(df, thresholds):
    if not thresholds or df.empty: return df
    out=df.copy()
    for th in thresholds:
        field, op, val = th['field'], th['op'], th['value']
        if field not in out.columns: continue
        if op=='>': out=out[out[field]>val]
        elif op=='>=': out=out[out[field]>=val]
        elif op=='<': out=out[out[field]<val]
        elif op=='<=': out=out[out[field]<=val]
    return out

def _compute_growth_clients(df_clienti, df_ordini, basis='fatturato', anni_list=None, anni_range=None,
                            paese=None, regione=None, min_start=5000.0, top_n=3, prefer='cagr'):
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if not allowed: return pd.DataFrame(), "Nessun cliente nell'area/periodo richiesto."
    if basis=='fatturato':
        dfc = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
        if anni_list: dfc = dfc[dfc['ANNO'].isin(anni_list)]
        if anni_range: dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
        series = dfc.groupby(['CLIENTE','ANNO'])['FATTURATO'].sum().reset_index().rename(columns={'FATTURATO':'VAL'})
    else:
        dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
        if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
        if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
        series = dfo.groupby(['nome_cliente','ANNO'])['KG'].sum().reset_index().rename(columns={'nome_cliente':'CLIENTE','KG':'VAL'})
    if series.empty: return pd.DataFrame(), "Nessun dato disponibile per il calcolo della crescita."
    def _growth_for_cli(df_cli):
        df_cli = df_cli.sort_values('ANNO')
        valid = df_cli[df_cli['VAL']>0]
        if len(valid)<2: return None
        a0,v0 = valid.iloc[0]['ANNO'], float(valid.iloc[0]['VAL'])
        a1,v1 = valid.iloc[-1]['ANNO'], float(valid.iloc[-1]['VAL'])
        n = max(1, int(a1)-int(a0))
        delta = v1 - v0
        cagr = (v1/v0)**(1/n) - 1 if v0>0 and n>0 else np.nan
        return pd.Series({'Cliente':df_cli['CLIENTE'].iloc[0],'Anno Inizio':a0,'Valore Inizio':v0,
                          'Anno Fine':a1,'Valore Fine':v1,'Δ Assoluto':delta,'CAGR %':cagr*100})
    res = series.groupby('CLIENTE').apply(_growth_for_cli).dropna().reset_index(drop=True)
    if res.empty: return pd.DataFrame(), "Nessun cliente con almeno 2 anni validi."
    res = res.sort_values('CAGR %' if prefer=='cagr' else 'Δ Assoluto', ascending=False).head(top_n).copy()
    if basis=='fatturato':
        res['Valore Inizio'] = res['Valore Inizio'].apply(format_euro_robust)
        res['Valore Fine'] = res['Valore Fine'].apply(format_euro_robust)
        res['Δ Assoluto'] = res['Δ Assoluto'].apply(format_euro_robust)
    else:
        res['Valore Inizio'] = res['Valore Inizio'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
        res['Valore Fine'] = res['Valore Fine'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
        res['Δ Assoluto'] = res['Δ Assoluto'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
    res['CAGR %'] = res['CAGR %'].map(lambda x: f"{x:.2f}%")
    res['Cliente'] = res['Cliente'].str.upper()
    return res, None

def _compute_share(df_clienti, df_ordini, entity='clienti', basis='fatturato',
                   anni_list=None, anni_range=None, paese=None, regione=None, top_n=3):
    if basis=='fatturato':
        dfc_all = df_clienti.copy()
        if anni_list: dfc_all = dfc_all[dfc_all['ANNO'].isin(anni_list)]
        if anni_range: dfc_all = dfc_all[(dfc_all['ANNO'] >= anni_range['from']) & (dfc_all['ANNO'] <= anni_range['to'])]
        tot = float(dfc_all['FATTURATO'].sum())
    else:
        dfo_all = df_ordini.copy()
        if anni_list: dfo_all = dfo_all[dfo_all['ANNO'].isin(anni_list)]
        if anni_range: dfo_all = dfo_all[(dfo_all['ANNO'] >= anni_range['from']) & (dfo_all['ANNO'] <= anni_range['to'])]
        tot = float(dfo_all['KG'].sum())
    if tot<=0: return pd.DataFrame(), "Totale nullo: impossibile calcolare la quota."
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if basis=='fatturato':
        dfc_sub = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
        if anni_list: dfc_sub = dfc_sub[dfc_sub['ANNO'].isin(anni_list)]
        if anni_range: dfc_sub = dfc_sub[(dfc_sub['ANNO'] >= anni_range['from']) & (dfc_sub['ANNO'] <= anni_range['to'])]
        sub_val = float(dfc_sub['FATTURATO'].sum())
    else:
        dfo_sub = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
        if anni_list: dfo_sub = dfo_sub[dfo_sub['ANNO'].isin(anni_list)]
        if anni_range: dfo_sub = dfo_sub[(dfo_sub['ANNO'] >= anni_range['from']) & (dfo_sub['ANNO'] <= anni_range['to'])]
        sub_val = float(dfo_sub['KG'].sum())
    quota = sub_val/tot*100.0
    if basis=='fatturato':
        view = pd.DataFrame({'Totale (€)':[format_euro_robust(tot)], 'Sottoinsieme (€)':[format_euro_robust(sub_val)], 'Quota %':[f"{quota:.2f}%"]})
    else:
        view = pd.DataFrame({'Totale (Kg)':[f"{tot:,.2f}".replace(",", ".")], 'Sottoinsieme (Kg)':[f"{sub_val:,.2f}".replace(",", ".")], 'Quota %':[f"{quota:.2f}%"]})
    return view, None

def _compute_ranking(df_clienti, df_ordini, entity='clienti', metric='fatturato',
                     anni_list=None, anni_range=None, paese=None, regione=None,
                     thresholds=None, top_n=3):
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if entity=='clienti':
        dfc = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
        if anni_list: dfc = dfc[dfc['ANNO'].isin(anni_list)]
        if anni_range: dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
        fatt = dfc.groupby('CLIENTE', as_index=False)['FATTURATO'].sum()
        dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
        if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
        if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
        kg = dfo.groupby('nome_cliente', as_index=False)['KG'].sum()
        fatt_o = dfo.groupby('nome_cliente', as_index=False)['FATTURATO_ORDINE'].sum()
        base = pd.merge(fatt, kg, left_on='CLIENTE', right_on='nome_cliente', how='outer')
        base = pd.merge(base, fatt_o, on='nome_cliente', how='outer')
        base['CLIENTE'] = base['CLIENTE'].fillna(base['nome_cliente'])
        base.drop(columns=['nome_cliente'], inplace=True)
        base[['FATTURATO','KG','FATTURATO_ORDINE']] = base[['FATTURATO','KG','FATTURATO_ORDINE']].fillna(0.0)
        if metric=='fatturato': base['METRICA']=base['FATTURATO']
        elif metric=='kg': base['METRICA']=base['KG']
        else: base['METRICA']=np.where(base['KG']>0, base['FATTURATO_ORDINE']/base['KG'], np.nan)
        base.rename(columns={'FATTURATO':'fatturato','KG':'kg'}, inplace=True)
        base = _apply_thresholds(base, thresholds).dropna(subset=['METRICA'])
        out = base.sort_values('METRICA', ascending=False).head(top_n).copy()
        if out.empty: return pd.DataFrame(), "Nessun risultato dopo i filtri/soglie."
        disp = pd.DataFrame({
            'CLIENTE': out['CLIENTE'].str.upper(),
            'Fatturato Totale': out['fatturato'].apply(format_euro_robust),
            'Kg Totali': out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", ".")),
            'Fatturato Ordini': out['FATTURATO_ORDINE'].apply(format_euro_robust)
        })
        if metric=='fatturato':
            disp['Metrica'] = out['fatturato'].apply(format_euro_robust)
        elif metric=='kg':
            disp['Metrica'] = out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
        else:
            disp['€/Kg medio'] = out['METRICA'].apply(format_euro_robust)
        return disp, None
    dfo = df_ordini.copy()
    if allowed: dfo = dfo[dfo['nome_cliente'].isin(allowed)]
    if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
    if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
    grp = 'ARTICOLO' if entity=='articoli' else 'COLORE'
    agg = dfo.groupby(grp, as_index=False).agg(kg=('KG','sum'), fatturato=('FATTURATO_ORDINE','sum'))
    if metric=='fatturato': agg['METRICA']=agg['fatturato']
    elif metric=='kg': agg['METRICA']=agg['kg']
    else: agg['METRICA']=np.where(agg['kg']>0, agg['fatturato']/agg['kg'], np.nan)
    agg = _apply_thresholds(agg, thresholds).dropna(subset=['METRICA'])
    out = agg.sort_values('METRICA', ascending=False).head(top_n).copy()
    if out.empty: return pd.DataFrame(), "Nessun risultato dopo i filtri/soglie."
    disp = pd.DataFrame({
        grp.upper(): out[grp],
        'Kg Totali': out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", ".")),
        'Fatturato': out['fatturato'].apply(format_euro_robust)
    })
    if metric=='fatturato':
        disp['Metrica'] = out['fatturato'].apply(format_euro_robust)
    elif metric=='kg':
        disp['Metrica'] = out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
    else:
        disp['€/Kg medio'] = out['METRICA'].apply(format_euro_robust)
    return disp, None

def _compute_avg(df_clienti, df_ordini, entity='clienti', metric='fatturato', anni_list=None, anni_range=None,
                 paese=None, regione=None, how='mean'):
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if entity=='clienti':
        if metric=='fatturato':
            dfc = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
            if anni_list: dfc = dfc[dfc['ANNO'].isin(anni_list)]
            if anni_range: dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
            agg = dfc.groupby('CLIENTE', as_index=False)['FATTURATO'].sum().rename(columns={'FATTURATO':'val'})
        elif metric=='kg':
            dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
            if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
            if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
            agg = dfo.groupby('nome_cliente', as_index=False)['KG'].sum().rename(columns={'nome_cliente':'CLIENTE','KG':'val'})
        else:
            dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
            if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
            if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
            by = dfo.groupby('nome_cliente', as_index=False).agg(Fatt=('FATTURATO_ORDINE','sum'), Kg=('KG','sum'))
            by['val'] = np.where(by['Kg']>0, by['Fatt']/by['Kg'], np.nan)
            agg = by.rename(columns={'nome_cliente':'CLIENTE'})[['CLIENTE','val']].dropna()
        if agg.empty: return pd.DataFrame(), "Nessun dato per il calcolo."
        val = float(agg['val'].median()) if how=='median' else float(agg['val'].mean())
        if metric=='fatturato': view=pd.DataFrame({'Valore':[format_euro_robust(val)], 'Metodo':[how]})
        elif metric=='kg': view=pd.DataFrame({'Valore':[f"{val:,.2f} Kg".replace(',', '.')], 'Metodo':[how]})
        else: view=pd.DataFrame({'Valore':[format_euro_robust(val)], 'Metodo':[how]})
        return view, None
    dfo = df_ordini.copy()
    if allowed: dfo = dfo[dfo['nome_cliente'].isin(allowed)]
    if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
    if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
    grp = 'ARTICOLO' if entity=='articoli' else 'COLORE'
    if metric=='fatturato':
        agg = dfo.groupby(grp, as_index=False)['FATTURATO_ORDINE'].sum().rename(columns={'FATTURATO_ORDINE':'val'})
    elif metric=='kg':
        agg = dfo.groupby(grp, as_index=False)['KG'].sum().rename(columns={'KG':'val'})
    else:
        by = dfo.groupby(grp, as_index=False).agg(Fatt=('FATTURATO_ORDINE','sum'), Kg=('KG','sum'))
        by['val'] = np.where(by['Kg']>0, by['Fatt']/by['Kg'], np.nan)
        agg = by[[grp,'val']].dropna()
    if agg.empty: return pd.DataFrame(), "Nessun dato per il calcolo."
    val = float(agg['val'].median()) if how=='median' else float(agg['val'].mean())
    if metric=='fatturato': view=pd.DataFrame({'Valore':[format_euro_robust(val)], 'Metodo':[how]})
    elif metric=='kg': view=pd.DataFrame({'Valore':[f"{val:,.2f} Kg".replace(',', '.')], 'Metodo':[how]})
    else: view=pd.DataFrame({'Valore':[format_euro_robust(val)], 'Metodo':[how]})
    return view, None

def compute_copilot_answer(df_clienti, df_ordini, intent):
    anni_list = intent.get('anni') or None
    anni_range = intent.get('anni_range')
    paese = intent.get('paese')
    regione = intent.get('regione')
    entity = intent.get('entity') or 'clienti'
    metric = intent.get('metric') or ('fatturato')
    top_n = intent.get('top_n') or 3
    thresholds = intent.get('thresholds') or []
    if intent.get('intent')=='growth':
        basis = 'kg' if metric=='kg' else 'fatturato'
        res, err = _compute_growth_clients(df_clienti, df_ordini, basis=basis, anni_list=anni_list, anni_range=anni_range,
                                           paese=paese, regione=regione, min_start=5000.0 if basis=='fatturato' else 500.0,
                                           top_n=top_n, prefer='cagr')
        if err: return {'table':pd.DataFrame(), 'explain':err, 'assumption':intent}
        explain = f"Crescita calcolata come **CAGR** (e Δ). Periodo: **{anni_range or ', '.join(anni_list or ['tutti'])}** su **{basis}**."
        return {'table':res, 'explain':explain, 'assumption':intent}
    if intent.get('intent')=='share':
        basis = 'kg' if metric=='kg' else 'fatturato'
        res, err = _compute_share(df_clienti, df_ordini, entity=entity, basis=basis, anni_list=anni_list, anni_range=anni_range,
                                  paese=paese, regione=regione, top_n=top_n)
        if err: return {'table':pd.DataFrame(), 'explain':err, 'assumption':intent}
        explain = f"Quota = sottoinsieme/totale su **{basis}**."
        return {'table':res, 'explain':explain, 'assumption':intent}
    if intent.get('intent') in ('mean','median'):
        how = 'median' if intent.get('intent')=='median' else 'mean'
        res, err = _compute_avg(df_clienti, df_ordini, entity=entity, metric=metric, anni_list=anni_list, anni_range=anni_range,
                                paese=paese, regione=regione, how=how)
        if err: return {'table':pd.DataFrame(), 'explain':err, 'assumption':intent}
        explain = f"{'Mediana' if how=='median' else 'Media'} per **{entity}** su **{metric}**."
        return {'table':res, 'explain':explain, 'assumption':intent}
    res, err = _compute_ranking(df_clienti, df_ordini, entity=entity, metric=metric, anni_list=anni_list, anni_range=anni_range,
                                paese=paese, regione=regione, thresholds=thresholds, top_n=top_n)
    if err: return {'table':pd.DataFrame(), 'explain':err, 'assumption':intent}
    explain = f"Ranking Top {top_n} per **{entity}** su **{metric}**."
    return {'table':res, 'explain':explain, 'assumption':intent}

# ===================== PAGINE =====================
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
        ordini_filtrati = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
        if analysis_mode == "Aggrega Anni":
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown("###### Top 10 Articoli (Kg)")
                st.dataframe(ordini_filtrati.groupby('ARTICOLO')['KG'].sum().nlargest(10).reset_index(), use_container_width=True, hide_index=True, height=385)
            with c2:
                st.markdown("###### Top 10 Colori (Kg)")
                st.dataframe(ordini_filtrati.groupby('COLORE')['KG'].sum().nlargest(10).reset_index(), use_container_width=True, hide_index=True, height=385)
            with c3:
                st.markdown("###### Top 10 Articolo-Colore (Kg)")
                st.dataframe(ordini_filtrati.groupby(['ARTICOLO','COLORE'])['KG'].sum().nlargest(10).reset_index(), use_container_width=True, hide_index=True, height=385)
        else:
            st.info("Modalità Confronto Anni: top articoli per anno.")
            top10 = ordini_filtrati.groupby('ARTICOLO')['KG'].sum().nlargest(10).index
            dfc = ordini_filtrati[ordini_filtrati['ARTICOLO'].isin(top10)]
            pivot = dfc.pivot_table(index='ARTICOLO', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
            pivot['Totale'] = pivot.sum(axis=1)
            pivot.sort_values('Totale', ascending=False, inplace=True)
            del pivot['Totale']
            st.dataframe(pivot.style.format("{:,.2f} Kg"), use_container_width=True)
    else:
        st.info("Seleziona almeno un anno per visualizzare i macrodati ordini.")

def page_elenco_clienti(df_clienti, df_ordini, anni_selezionati, paese_selezionato, analysis_mode):
    st.title("Elenco e Segmentazione Clienti")
    df_filtrato = df_clienti[df_clienti['ANNO'].isin(anni_selezionati)] if anni_selezionati else df_clienti
    if paese_selezionato != "Tutti":
        df_filtrato = df_filtrato[df_filtrato['PAESE']==paese_selezionato]

    st.subheader("Ranking Clienti")
    if not df_filtrato.empty:
        base = df_filtrato.groupby('CLIENTE').agg(PAESE=('PAESE','first')).reset_index()
        if analysis_mode == "Aggrega Anni":
            r = df_filtrato.groupby('CLIENTE').agg(Fatturato_Anagrafica=('FATTURATO','sum')).reset_index()
            if not df_ordini.empty and anni_selezionati:
                ordf = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
                ko = ordf.groupby('nome_cliente').agg(KG_Ordinati=('KG','sum')).reset_index()
                r = pd.merge(r, ko, left_on='CLIENTE', right_on='nome_cliente', how='left')
            r = pd.merge(r, base, on='CLIENTE', how='left')
            r['KG_Ordinati'] = r['KG_Ordinati'].fillna(0)
            r = r.sort_values('Fatturato_Anagrafica', ascending=False)
            r['CLIENTE_DISPLAY'] = r['CLIENTE'].str.upper()
            disp = r[['CLIENTE_DISPLAY','PAESE','Fatturato_Anagrafica','KG_Ordinati']].copy()
            disp.rename(columns={'CLIENTE_DISPLAY':'CLIENTE'}, inplace=True)
            disp['Fatturato_Anagrafica'] = disp['Fatturato_Anagrafica'].apply(format_euro_robust)
            disp['KG_Ordinati'] = disp['KG_Ordinati'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#","."))
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("Modalità Confronto Anni.")
            fatt = df_filtrato.pivot_table(index='CLIENTE', columns='ANNO', values='FATTURATO', aggfunc='sum')
            ordf = df_ordini[df_ordini['ANNO'].isin(anni_selezionati)]
            kgp = ordf.pivot_table(index='nome_cliente', columns='ANNO', values='KG', aggfunc='sum')
            comp = pd.merge(fatt, kgp, left_index=True, right_index=True, how='outer').fillna(0)
            cols=[]
            for y in sorted(anni_selezionati, reverse=True):
                cf, ck = f"Fatturato_{y}", f"KG_{y}"
                comp.rename(columns={y+'_x':cf, y+'_y':ck}, inplace=True, errors='ignore')
                if cf in comp.columns: comp[cf]=comp[cf].apply(format_euro_robust)
                if ck in comp.columns: comp[ck]=comp[ck].apply(lambda x: f"{x:,.2f} Kg")
                cols += [c for c in [cf,ck] if c in comp.columns]
            comp = pd.merge(comp, base, left_index=True, right_on='CLIENTE', how='left').set_index('CLIENTE')
            comp.reset_index(inplace=True)
            comp['CLIENTE'] = comp['CLIENTE'].str.upper()
            st.dataframe(comp[['CLIENTE','PAESE']+cols], use_container_width=True, hide_index=True)

def page_report_avanzati(df_ordini):
    st.title("Report Avanzati — Analisi Predittiva per Articolo")
    if df_ordini.empty or 'ARTICOLO' not in df_ordini.columns:
        st.info("Nessun dato ordini o 'ARTICOLO' assente."); return
    articoli = sorted([a for a in df_ordini['ARTICOLO'].dropna().unique() if str(a).strip()!=""])
    if not articoli: st.info("Nessun articolo disponibile."); return
    articolo_sel = st.selectbox("Seleziona un articolo", options=articoli, index=0)
    tmp = df_ordini[df_ordini['ARTICOLO']==articolo_sel].copy()
    tmp['ANNO_NUM'] = pd.to_numeric(tmp['ANNO'], errors='coerce')
    df_art = tmp.dropna(subset=['ANNO_NUM']).groupby('ANNO_NUM', as_index=False)['KG'].sum().sort_values('ANNO_NUM')
    if df_art.empty: st.warning("Nessun dato storico."); return
    x = df_art['ANNO_NUM'].values.astype(float); y = df_art['KG'].values.astype(float)
    if len(df_art)>=2:
        m,q = np.polyfit(x,y,1); x_line = np.linspace(x.min(), x.max()+1, 100); y_line = m*x_line + q
        trend = "In crescita" if m>0 else ("In calo" if m<0 else "Stabile")
    else:
        m,q = 0,y[0]; x_line,y_line = x,y; trend = "Dati insufficienti"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_art['ANNO_NUM'], y=df_art['KG'], mode='markers+lines', name='Storico Kg'))
    if len(df_art)>=2:
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Trend (regressione)', line=dict(dash='dash')))
        y_next = float(m*(np.max(x)+1) + q)
        fig.add_trace(go.Scatter(x=[np.max(x)+1], y=[y_next], mode='markers', name='Proiezione +1', marker=dict(symbol='diamond-open', size=10)))
    fig.update_layout(title=f"Andamento Kg — {articolo_sel}", xaxis_title="Anno", yaxis_title="Kg", hovermode="x unified")
    a,b,c = st.columns(3)
    a.metric("Anni coperti", f"{df_art['ANNO_NUM'].nunique()}"); b.metric("Kg totali", f"{df_art['KG'].sum():,.2f}".replace(",", ".")); c.metric("Tendenza", trend)
    st.plotly_chart(fig, use_container_width=True)
    if len(df_art)>=2: st.caption("Linea tratteggiata = regressione; rombo = proiezione anno successivo.")

def page_analisi_dettagliata(df_clienti, df_ordini, anni_disponibili, anni_selezionati_globali, analysis_mode):
    st.title("Analisi Dettagliata Cliente")
    # Nuovi clienti
    df_first = compute_first_year_per_client(df_clienti, df_ordini)
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        anno_new = st.selectbox("Anno 'nuovi clienti'", options=anni_disponibili, index=0, key="anno_nuovi")
    with c2:
        only_new = st.checkbox(f"Mostra solo nuovi clienti {anno_new}", value=False, key="flag_nuovi")
    with c3:
        all_cli = df_clienti['CLIENTE'].unique().tolist()
        if only_new:
            new_set = set(df_first[df_first['FIRST_YEAR']==str(anno_new)]['CLIENTE'])
            options = sorted([c.upper() for c in all_cli if c in new_set])
        else:
            options = sorted([c.upper() for c in all_cli])
        clienti_sel_up = st.multiselect("Seleziona clienti", options=options, key='client_selector_detail')

    if not clienti_sel_up:
        st.info("Seleziona uno o più clienti per iniziare.")
        return
    clienti_sel = [c.lower() for c in clienti_sel_up]

    anni_scheda = st.multiselect("Seleziona anni", options=anni_disponibili, default=anni_selezionati_globali, key="anni_dettaglio_selector")
    anno_rif = anni_scheda[0] if anni_scheda else anni_disponibili[0]

    # Azioni prioritarie
    st.subheader("📌 Azioni Prioritarie")
    df_actions = load_actions_table(anno_rif)
    if not df_actions.empty:
        df_actions['CLIENTE_DISPLAY'] = df_actions['cliente'].str.upper()
        todo = df_actions[df_actions['done']==0].copy()
        done = df_actions[df_actions['done']==1].copy()
        todo.sort_values(by=['priority','updated_at'], ascending=[False, False], inplace=True)
        done.sort_values(by=['updated_at'], ascending=False, inplace=True)
        col_todo, col_done = st.columns(2)
        with col_todo:
            st.markdown("**Da fare (ord. per priorità)**")
            if todo.empty: st.info("Nessuna azione in sospeso.")
            else:
                for _, r in todo.iterrows():
                    with st.expander(f"{r['CLIENTE_DISPLAY']} — ⭐ {int(r['stars'])} — Priorità {r['priority']:.2f}"):
                        note = st.text_area("Nota/Azione", value=r.get('note',''), key=f"note_todo_{r['cliente']}_{anno_rif}")
                        done_ck = st.checkbox("Segna come fatto", value=False, key=f"done_todo_{r['cliente']}_{anno_rif}")
                        if st.button("Salva", key=f"save_todo_{r['cliente']}_{anno_rif}"):
                            upsert_client_action(r['cliente'], anno_rif, note=note, done=1 if done_ck else 0)
                            st.success("Aggiornato."); st.rerun()
        with col_done:
            st.markdown("**Completate**")
            if done.empty: st.info("Nessuna azione completata.")
            else:
                for _, r in done.iterrows():
                    with st.expander(f"{r['CLIENTE_DISPLAY']} — ⭐ {int(r['stars'])} — Priorità {r['priority']:.2f}"):
                        note = st.text_area("Nota", value=r.get('note',''), key=f"note_done_{r['cliente']}_{anno_rif}")
                        undo = st.checkbox("Riporta in 'Da fare'", value=False, key=f"undo_done_{r['cliente']}_{anno_rif}")
                        if st.button("Aggiorna", key=f"save_done_{r['cliente']}_{anno_rif}"):
                            upsert_client_action(r['cliente'], anno_rif, note=note, done=0 if undo else 1)
                            st.success("Aggiornato."); st.rerun()
    else:
        st.info("Non ci sono azioni salvate per quest'anno. Verranno create dopo l'analisi IA.")

    st.header(f"Scheda Alleati: {', '.join(clienti_sel_up)}")
    tab_eval, tab_dati, tab_ordini = st.tabs(["Valutazione & IA","Anagrafica & Fatturato","Ordini & Statistiche"])

    with tab_eval:
        st.subheader(f"Valutazioni + Azioni (anno rif.: {anno_rif})")
        for cli in clienti_sel:
            with st.expander(cli.upper()):
                with st.form(key=f"evaluation_form_{cli}_{anno_rif}"):
                    ev = load_evaluation(cli, anno_rif)
                    cols = st.columns(3); tmp={}
                    for i,q in enumerate(EVALUATION_QUESTIONS):
                        with cols[i%3]:
                            tmp[q['key']] = st.slider(q['text'],1,5,value=ev.get(q['key'],3), key=f"{q['key']}_{cli}_{anno_rif}")
                    if st.form_submit_button("Salva Valutazione"):
                        save_evaluation(cli, anno_rif, tmp); st.cache_data.clear(); st.rerun()

                evc = load_evaluation(cli, anno_rif)
                tot, ve, vr = calculate_scores(evc)
                quad = get_matrix_quadrant(ve, vr)
                auto_stars = stars_from_total(tot)
                auto_prio = priority_from_quadrant(ve, vr)
                st.markdown(f"**Totale**: {tot:.0f}/75 • **Val.Economico**: {ve:.2f} • **Val.Relazionale**: {vr:.2f} • **Profilo**: _{quad}_")
                c1,c2,c3 = st.columns([1,1,2])
                with c1:
                    stars_sel = st.selectbox("Stelle (1–5)", options=[1,2,3,4,5], index=auto_stars-1, key=f"stars_{cli}_{anno_rif}")
                with c2:
                    st.metric("Priorità (auto)", f"{auto_prio:.2f}")
                with c3:
                    note_act = st.text_input("Nota/Azione", key=f"note_{cli}_{anno_rif}", placeholder="Es. visita/up-sell...")

                saved = load_last_ai_output(cli, anno_rif)
                if saved:
                    st.markdown("**Ultima analisi IA salvata**")
                    st.caption(f"Generata il: {saved['created_at']}")
                    st.markdown(saved['output'])
                else:
                    st.info("Nessuna analisi IA salvata per questo cliente/anno.")

                b1,b2 = st.columns(2)
                with b1:
                    if st.button("🔎 Analizza con IA", key=f"ai_{cli}_{anno_rif}"):
                        with st.spinner("Analisi IA..."):
                            digest = build_client_digest_anon(cli, anni_scheda, anno_rif, df_clienti, df_ordini)
                            system = ("Sei un business analyst per un'azienda B2B (nylon). Usa SOLO i dati forniti. "
                                      "Output in italiano, conciso, a punti: 1) Sintesi 2) Opportunità 3) Rischi 4) Azioni 30/60/90 gg.")
                            user = f"DATI CLIENTE (anonimizzati):\n{digest}"
                            out = call_llm(system, user)
                            st.markdown("#### Risultato IA"); st.markdown(out)
                            save_ai_output(cli, anno_rif, digest, out)
                            upsert_client_action(cli, anno_rif, stars=auto_stars, priority=auto_prio)
                            st.rerun()
                with b2:
                    if st.button("💾 Salva Stelle/Priorità/Azione", key=f"save_{cli}_{anno_rif}"):
                        upsert_client_action(cli, anno_rif, stars=stars_sel, priority=auto_prio, note=note_act, done=0)
                        st.success("Azione salvata. Compare nella sezione 'Da fare'."); st.rerun()

    with tab_dati:
        for cli in clienti_sel:
            with st.expander(f"Dati per {cli.upper()}"):
                dati = df_clienti[df_clienti['CLIENTE']==cli]
                st.subheader(f"Anagrafica (anno rif.: {anno_rif})")
                anag = dati[dati['ANNO']==anno_rif]
                if not anag.empty:
                    row = anag.iloc[0]
                elif not dati.empty:
                    row = dati.sort_values('ANNO', ascending=False).iloc[0]
                    st.info(f"Dati anagrafici {anno_rif} non trovati: mostro i più recenti.")
                else:
                    st.warning("Dati anagrafici non disponibili."); continue
                cc = st.columns(3)
                cc[0].markdown(f"**Indirizzo:**<br>{row.get('VIA','N/D')}", unsafe_allow_html=True)
                cc[1].markdown(f"**Paese:**<br>{row.get('PAESE','N/D')}", unsafe_allow_html=True)
                cc[2].markdown(f"**Contatti:**<br>Titolare: {row.get('TITOLARE','N/D')}", unsafe_allow_html=True)
                st.divider(); st.subheader("Andamento Fatturato Annuale")
                f_ann = dati.groupby('ANNO')['FATTURATO'].sum().sort_index()
                fig = go.Figure(data=[go.Bar(x=f_ann.index, y=f_ann.values, text=[format_euro_robust(v) for v in f_ann.values], textposition='auto')])
                st.plotly_chart(fig, use_container_width=True)

    with tab_ordini:
        st.subheader(f"Statistiche Ordini (anni: {', '.join(anni_scheda)})")
        if df_ordini.empty or not anni_scheda:
            st.info("Seleziona almeno un anno.")
        else:
            ordsel = df_ordini[(df_ordini['ANNO'].isin(anni_scheda)) & (df_ordini['nome_cliente'].isin(clienti_sel))]
            if ordsel.empty:
                st.info("Nessun ordine per il periodo.")
            elif analysis_mode=="Aggrega Anni":
                st.subheader("Statistiche Aggregate")
                kg_tot = ordsel['KG'].sum(); f_tot = ordsel['FATTURATO_ORDINE'].sum()
                pmedio = (f_tot/kg_tot) if kg_tot>0 else 0
                a,b,c,d = st.columns(4)
                a.metric("Kg Totali", f"{kg_tot:,.2f} Kg".replace(",", "."))
                b.metric("Fatturato Ordini", format_euro_robust(f_tot))
                c.metric("Prezzo Medio Kg", f"{format_euro_robust(pmedio)} /Kg")
                d.metric("Righe Ordine", f"{len(ordsel)}")
                st.divider(); st.subheader("Dettaglio per Cliente")
                for cli in clienti_sel:
                    with st.expander(f"Ordini per {cli.upper()}"):
                        oc = ordsel[ordsel['nome_cliente']==cli]
                        if oc.empty: st.write("Nessun dato."); continue
                        kg = oc['KG'].sum(); fa = oc['FATTURATO_ORDINE'].sum()
                        pm = (fa/kg) if kg>0 else 0
                        k1,k2,k3,k4 = st.columns(4)
                        k1.metric("Kg Totali", f"{kg:,.2f} Kg".replace(",", "."))
                        k2.metric("Fatturato", format_euro_robust(fa))
                        k3.metric("€/Kg", f"{format_euro_robust(pm)} /Kg")
                        k4.metric("Righe", f"{len(oc)}")
                        c1,c2,c3 = st.columns(3)
                        top_art = oc.groupby('ARTICOLO', dropna=False)['KG'].sum().sort_values(ascending=False).head(10).reset_index()
                        c1.dataframe(top_art.rename(columns={'KG':'KG Totali'}), use_container_width=True, hide_index=True)
                        top_col = oc.groupby('COLORE', dropna=False)['KG'].sum().sort_values(ascending=False).head(10).reset_index()
                        c2.dataframe(top_col.rename(columns={'KG':'KG Totali'}), use_container_width=True, hide_index=True)
                        top_combo = oc.groupby(['ARTICOLO','COLORE'], dropna=False)['KG'].sum().sort_values(ascending=False).head(10).reset_index()
                        c3.dataframe(top_combo.rename(columns={'KG':'KG Totali'}), use_container_width=True, hide_index=True)
                        st.markdown("##### €/Kg per Articolo")
                        by_art = oc.groupby('ARTICOLO', dropna=False).agg(KG=('KG','sum'), Fatturato=('FATTURATO_ORDINE','sum')).reset_index()
                        by_art['€/Kg'] = np.where(by_art['KG']>0, by_art['Fatturato']/by_art['KG'], 0)
                        by_art = by_art.sort_values('€/Kg', ascending=False).head(15)
                        disp = by_art[['ARTICOLO','KG','Fatturato','€/Kg']].copy()
                        disp['KG'] = disp['KG'].apply(lambda x: f"{x:,.2f} Kg".replace(",", "#").replace(".", ",").replace("#","."))
                        disp['Fatturato'] = disp['Fatturato'].apply(format_euro_robust)
                        disp['€/Kg'] = disp['€/Kg'].apply(format_euro_robust)
                        st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("Confronto anni.")
                for cli in clienti_sel:
                    with st.expander(f"Confronto per {cli.upper()}"):
                        oc = ordsel[ordsel['nome_cliente']==cli]
                        if oc.empty: st.write("Nessun dato."); continue
                        st.markdown("##### Articolo")
                        p1 = oc.pivot_table(index='ARTICOLO', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
                        p1['Totale'] = p1.sum(axis=1)
                        st.dataframe(p1.sort_values('Totale', ascending=False).drop(columns='Totale').style.format("{:,.2f}"), use_container_width=True)
                        st.markdown("##### Colore")
                        p2 = oc.pivot_table(index='COLORE', columns='ANNO', values='KG', aggfunc='sum').fillna(0)
                        p2['Totale'] = p2.sum(axis=1)
                        st.dataframe(p2.sort_values('Totale', ascending=False).drop(columns='Totale').style.format("{:,.2f}"), use_container_width=True)

def page_copilot(df_clienti, df_ordini, anni_disponibili):
    st.title("🤖 Copilot IA — Query Libera su Tutta la Dashboard")
    st.caption("Capisce crescita (CAGR/Δ), quote %, medie/mediane, soglie (> / <), ranking. Svincolato dai filtri globali.")
    q = st.text_input("Scrivi la tua domanda", key="copilot_q")
    add_ai_comment = st.checkbox("Aggiungi commento IA (opzionale)", value=False)
    if st.button("Esegui"):
        if not q.strip(): st.warning("Inserisci una domanda."); return
        intent = parse_user_query(q)
        res = compute_copilot_answer(df_clienti, df_ordini, intent)
        st.markdown(f"**Interpretazione automatica:** `{json.dumps(intent, ensure_ascii=False)}`")
        if res["table"].empty:
            st.error("Nessun risultato con i dati disponibili."); return
        st.markdown(res["explain"])
        st.dataframe(res["table"], use_container_width=True, hide_index=True)
        if add_ai_comment:
            sample = res["table"].head(5).to_dict(orient="records")
            system = ("Sei un assistente analitico per una dashboard B2B. "
                      "Commenta in 5-6 bullet in italiano: insight, caveat, prossime azioni.")
            user = f"Domanda: {q}\nInterpretazione: {json.dumps(intent, ensure_ascii=False)}\nTabella (estratto): {sample}"
            ai_out = call_llm(system, user)
            st.markdown("**Commento IA:**")
            st.markdown(ai_out)

def page_stato_dati(df_clienti, df_ordini):
    st.title("Stato dei Dati e Diagnostica")
    st.header("1) File trovati in data/")
    files = [p.name for p in DATA_DIR.glob('*')]
    st.dataframe(files if files else ["(vuoto)"], use_container_width=True)
    st.header("2) Caricamento")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Anagrafica")
        if not df_clienti.empty:
            st.metric("Righe (post esplosione anni)", len(df_clienti))
            st.metric("Clienti unici", df_clienti['CLIENTE'].nunique())
        else: st.warning("Anagrafica mancante/vuota.")
    with c2:
        st.subheader("Ordini")
        if not df_ordini.empty:
            st.metric("Righe totali", len(df_ordini))
            st.metric("Clienti (ordini)", df_ordini['nome_cliente'].nunique())
        else: st.warning("Ordini mancanti/vuoti.")
    st.header("3) Corrispondenze")
    if not df_clienti.empty and not df_ordini.empty:
        ca = set(df_clienti['CLIENTE'].unique())
        co = set(df_ordini['nome_cliente'].unique())
        inter = ca.intersection(co); orfani = co - ca
        st.metric("Clienti corrispondenti", len(inter))
        if orfani:
            st.error(f"{len(orfani)} clienti negli ordini non presenti in anagrafica.")
            st.dataframe(sorted([c.upper() for c in orfani]), use_container_width=True)
        else:
            st.success("Tutti i clienti ordini corrispondono all'anagrafica.")
    else:
        st.info("Servono sia anagrafica che ordini per la diagnosi.")

# ===================== BOOTSTRAP =====================
init_db()
df_clienti = load_clients_df()
df_ordini = load_all_orders_df()
if df_clienti.empty and df_ordini.empty:
    st.error("Nessun file dati ('elenco clienti.csv' o 'ordini_*') in data/.")
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
