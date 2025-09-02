# =========================
# 🚀 COPILOT — FUNZIONI NUOVE
# Sostituisci/aggiungi queste funzioni nel file principale
# (richiedono: pandas as pd, numpy as np, re, json, REGIONE_TO_PROV,
#  format_euro_robust, apply_region_filter già presenti)
# =========================

# ---------- PARSER INTENTI ----------
def parse_user_query(q: str):
    """
    Estrae intent dall'italiano naturale:
    - intent: 'growth' | 'share' | 'mean' | 'median' | 'ranking'
    - metric: 'fatturato' | 'kg' | 'euro_kg'
    - entity: 'clienti' | 'articoli' | 'colori'
    - anni: elenco anni espliciti (['2020','2021']) o range {'from':'2019','to':'2023'}
    - top_n: int (default 3 se ranking)
    - paese: 'Italia' | 'Estero' | None
    - regione: key in REGIONE_TO_PROV
    - thresholds: lista di condizioni [{'field':'kg'|'fatturato'|'euro_kg','op':'>=','value':3000.0}]
    """
    txt = q.lower()

    # --- intent base ---
    intent = 'ranking'
    if re.search(r'\bcresc|aument|trend|variaz|cagr|yoy\b', txt):
        intent = 'growth'
    elif re.search(r'\bquota|percentual|%\b', txt):
        intent = 'share'
    elif re.search(r'\bmediana\b', txt):
        intent = 'median'
    elif re.search(r'\bmedio|media|average\b', txt):
        intent = 'mean'

    # --- metrica/basis ---
    metric = None
    if re.search(r'€/kg|euro/kg|profittevol', txt):
        metric = 'euro_kg'
    elif re.search(r'\bkg\b|\bquantit', txt):
        metric = 'kg'
    elif re.search(r'fatturat', txt):
        metric = 'fatturato'

    # --- entità ---
    entity = 'clienti'
    if 'articol' in txt: entity = 'articoli'
    elif 'color' in txt: entity = 'colori'

    # --- anni espliciti o range ---
    years = re.findall(r'\b(20\d{2})\b', txt)
    anni = list(dict.fromkeys(years))  # unique
    range_match = re.search(r'(dal|da|tra|fra)\s*(20\d{2})\s*(al|a|e)\s*(20\d{2})', txt)
    anni_range = None
    if range_match:
        a1, a2 = range_match.group(2), range_match.group(4)
        if a1 <= a2:
            anni_range = {'from': a1, 'to': a2}
    # "tutti gli anni" -> segnalo None (userà range completo dati)
    if re.search(r'tutt[ioa] gli anni|tutti i periodi', txt):
        anni, anni_range = [], None

    # --- top N ---
    top_n = None
    m = re.search(r'\btop\s+(\d+)\b', txt) or re.search(r'\b(primi|migliori?)\s+(\d+)\b', txt)
    if m: top_n = int(m.groups()[-1])
    elif re.search(r'\bpiù|massim|maggior|rank 1\b', txt): top_n = 1

    # --- area (paese/regioni) ---
    paese = 'Italia' if 'italia' in txt else ('Estero' if 'estero' in txt else None)
    regione = None
    for reg in REGIONE_TO_PROV.keys():
        if reg in txt:
            regione = reg
            break

    # --- soglie (>, <, >=, <=) con euro/kg opzionali ---
    thresholds = []
    # pattern: valore (con , o .) eventualmente con € o 'kg'
    for pat in [r'(>=|<=|>|<)\s*€?\s*([\d\.,]+)\s*(€/kg|euro/kg|kg)?',
                r'(sopra|oltre|maggiore di)\s*€?\s*([\d\.,]+)\s*(€/kg|euro/kg|kg)?',
                r'(sotto|inferiore a|minore di)\s*€?\s*([\d\.,]+)\s*(€/kg|euro/kg|kg)?']:
        for m in re.finditer(pat, txt):
            op_raw = m.group(1)
            val_raw = m.group(2).replace('.', '').replace(',', '.')
            unit = m.group(3) or ''
            try:
                value = float(val_raw)
            except:
                continue
            if op_raw in ('sopra','oltre','maggiore di'): op = '>'
            elif op_raw in ('sotto','inferiore a','minore di'): op = '<'
            else: op = op_raw
            field = 'fatturato'
            if 'kg' in unit: field = 'kg'
            if '€/kg' in unit or 'euro/kg' in unit: field = 'euro_kg'
            thresholds.append({'field': field, 'op': op, 'value': value})

    return {
        'intent': intent,
        'metric': metric,
        'entity': entity,
        'anni': anni,            # es. ['2021','2022']
        'anni_range': anni_range,# es. {'from':'2019','to':'2023'}
        'top_n': top_n,
        'paese': paese,
        'regione': regione,
        'thresholds': thresholds
    }

# ---------- HELPER FILTRI AREA ----------
def _allowed_clients_by_area(df_clienti, anni_list=None, anni_range=None, paese=None, regione=None):
    dfc = df_clienti.copy()
    if anni_list:
        dfc = dfc[dfc['ANNO'].isin(anni_list)]
    if anni_range:
        dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
    if paese:
        dfc = dfc[dfc['PAESE'] == paese]
    if regione:
        provs = REGIONE_TO_PROV.get(regione, set())
        if 'PROVINCIA' in dfc.columns and provs:
            dfc = dfc[dfc['PROVINCIA'].str.upper().isin(provs)]
    return set(dfc['CLIENTE'])

# ---------- HELPER SOGLIE ----------
def _apply_thresholds(df, thresholds):
    if not thresholds or df.empty: return df
    out = df.copy()
    for th in thresholds:
        field, op, val = th['field'], th['op'], th['value']
        if field not in out.columns: continue
        if op == '>': out = out[out[field] > val]
        elif op == '>=': out = out[out[field] >= val]
        elif op == '<': out = out[out[field] < val]
        elif op == '<=': out = out[out[field] <= val]
    return out

# ---------- GROWTH (Δ e CAGR) PER CLIENTE ----------
def _compute_growth_clients(df_clienti, df_ordini, basis='fatturato', anni_list=None, anni_range=None,
                            paese=None, regione=None, min_start=5000.0, top_n=3, prefer='cagr'):
    """
    basis: 'fatturato' | 'kg'
    prefer: 'cagr' | 'delta'
    """
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if not allowed:
        return pd.DataFrame(), "Nessun cliente nell'area/periodo richiesto."

    # Serie annuale per cliente
    # fatturato da df_clienti, kg da df_ordini
    if basis == 'fatturato':
        dfc = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
        if anni_list:
            dfc = dfc[dfc['ANNO'].isin(anni_list)]
        if anni_range:
            dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
        series = dfc.groupby(['CLIENTE','ANNO'])['FATTURATO'].sum().reset_index()
        val_col = 'VAL'
    else:  # kg
        dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
        if anni_list:
            dfo = dfo[dfo['ANNO'].isin(anni_list)]
        if anni_range:
            dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
        series = dfo.groupby(['nome_cliente','ANNO'])['KG'].sum().reset_index().rename(columns={'nome_cliente':'CLIENTE'})
        val_col = 'VAL'

    if series.empty:
        return pd.DataFrame(), "Nessun dato disponibile per il calcolo della crescita."

    series = series.rename(columns={'FATTURATO': val_col})
    # Calcolo primo e ultimo anno non nulli per ciascun cliente
    def _growth_for_cli(df_cli):
        df_cli = df_cli.sort_values('ANNO')
        # prendo primo e ultimo anno con valore > 0
        valid = df_cli[df_cli[val_col] > 0]
        if len(valid) < 2:
            return None
        a0, v0 = valid.iloc[0]['ANNO'], float(valid.iloc[0][val_col])
        a1, v1 = valid.iloc[-1]['ANNO'], float(valid.iloc[-1][val_col])
        if v0 < (min_start if basis=='fatturato' else (min_start if basis=='kg' else 0)):
            # soglia minima sul valore iniziale per robustezza
            pass
        n_years = max(1, int(a1) - int(a0))
        delta = v1 - v0
        cagr = (v1 / v0) ** (1 / n_years) - 1 if v0 > 0 and n_years > 0 else np.nan
        return pd.Series({'Cliente': df_cli['CLIENTE'].iloc[0], 'Anno Inizio': a0, 'Valore Inizio': v0,
                          'Anno Fine': a1, 'Valore Fine': v1, 'Δ Assoluto': delta, 'CAGR %': cagr * 100})
    res = series.groupby('CLIENTE').apply(_growth_for_cli).dropna().reset_index(drop=True)
    if res.empty:
        return pd.DataFrame(), "Nessun cliente con almeno 2 anni validi."

    # Ordinamento
    if prefer == 'cagr':
        res = res.sort_values('CAGR %', ascending=False)
    else:
        res = res.sort_values('Δ Assoluto', ascending=False)
    res = res.head(top_n).copy()

    # formattazione
    if basis == 'fatturato':
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

# ---------- QUOTE % ----------
def _compute_share(df_clienti, df_ordini, entity='clienti', basis='fatturato',
                   anni_list=None, anni_range=None, paese=None, regione=None, top_n=3):
    """
    Calcola quota percentuale di un sottoinsieme (paese/regione) sul totale,
    e opzionalmente ranking per entità.
    """
    # Base totale (senza filtro area)
    if basis == 'fatturato':
        dfc_all = df_clienti.copy()
        if anni_list: dfc_all = dfc_all[dfc_all['ANNO'].isin(anni_list)]
        if anni_range: dfc_all = dfc_all[(dfc_all['ANNO'] >= anni_range['from']) & (dfc_all['ANNO'] <= anni_range['to'])]
        tot = float(dfc_all['FATTURATO'].sum())
    else:  # kg
        dfo_all = df_ordini.copy()
        if anni_list: dfo_all = dfo_all[dfo_all['ANNO'].isin(anni_list)]
        if anni_range: dfo_all = dfo_all[(dfo_all['ANNO'] >= anni_range['from']) & (dfo_all['ANNO'] <= anni_range['to'])]
        tot = float(dfo_all['KG'].sum())

    if tot <= 0:
        return pd.DataFrame(), "Totale nullo: impossibile calcolare la quota."

    # Sottoinsieme (area)
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if basis == 'fatturato':
        dfc_sub = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
        if anni_list: dfc_sub = dfc_sub[dfc_sub['ANNO'].isin(anni_list)]
        if anni_range: dfc_sub = dfc_sub[(dfc_sub['ANNO'] >= anni_range['from']) & (dfc_sub['ANNO'] <= anni_range['to'])]
        sub_val = float(dfc_sub['FATTURATO'].sum())
    else:
        dfo_sub = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
        if anni_list: dfo_sub = dfo_sub[dfo_sub['ANNO'].isin(anni_list)]
        if anni_range: dfo_sub = dfo_sub[(dfo_sub['ANNO'] >= anni_range['from']) & (dfo_sub['ANNO'] <= anni_range['to'])]
        sub_val = float(dfo_sub['KG'].sum())

    quota = sub_val / tot * 100.0

    # tabellina esplicativa
    if basis == 'fatturato':
        df_view = pd.DataFrame({
            'Totale (€)': [format_euro_robust(tot)],
            'Sottoinsieme (€)': [format_euro_robust(sub_val)],
            'Quota %': [f"{quota:.2f}%"]
        })
    else:
        df_view = pd.DataFrame({
            'Totale (Kg)': [f"{tot:,.2f}".replace(",", ".")],
            'Sottoinsieme (Kg)': [f"{sub_val:,.2f}".replace(",", ".")],
            'Quota %': [f"{quota:.2f}%"]
        })

    return df_view, None

# ---------- RANKING GENERALE (con soglie) ----------
def _compute_ranking(df_clienti, df_ordini, entity='clienti', metric='fatturato',
                     anni_list=None, anni_range=None, paese=None, regione=None,
                     thresholds=None, top_n=3):
    # Filtri area -> allowed clients
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)

    if entity == 'clienti':
        # fatturato da anagrafica
        dfc = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
        if anni_list: dfc = dfc[dfc['ANNO'].isin(anni_list)]
        if anni_range: dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
        fatt = dfc.groupby('CLIENTE', as_index=False)['FATTURATO'].sum()

        # ordini
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

        # metriche
        if metric == 'fatturato':
            base['METRICA'] = base['FATTURATO']
        elif metric == 'kg':
            base['METRICA'] = base['KG']
        else:  # euro_kg
            base['METRICA'] = np.where(base['KG']>0, base['FATTURATO_ORDINE']/base['KG'], np.nan)

        # soglie
        base.rename(columns={'FATTURATO':'fatturato','KG':'kg'}, inplace=True)
        base = _apply_thresholds(base, thresholds)
        base = base.dropna(subset=['METRICA'])

        # output
        out = base.sort_values('METRICA', ascending=False).head(top_n).copy()
        if out.empty:
            return pd.DataFrame(), "Nessun risultato dopo l'applicazione dei filtri/soglie."
        # format
        disp = pd.DataFrame({
            'CLIENTE': out['CLIENTE'].str.upper(),
            'Fatturato Totale': out['fatturato'].apply(format_euro_robust),
            'Kg Totali': out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", ".")),
            'Fatturato Ordini': out['FATTURATO_ORDINE'].apply(format_euro_robust)
        })
        if metric == 'fatturato':
            disp['Metrica'] = out['fatturato'].apply(format_euro_robust)
        elif metric == 'kg':
            disp['Metrica'] = out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
        else:
            disp['Metrica'] = out['METRICA'].apply(format_euro_robust).str.replace("€ ", "€ ")
            disp.rename(columns={'Metrica':'€/Kg medio'}, inplace=True)
        return disp, None

    # --- articoli o colori ---
    dfo = df_ordini.copy()
    # limita ai clienti allowed (area)
    if allowed: dfo = dfo[dfo['nome_cliente'].isin(allowed)]
    if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
    if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
    grp = 'ARTICOLO' if entity=='articoli' else 'COLORE'
    agg = dfo.groupby(grp, as_index=False).agg(kg=('KG','sum'), fatturato=('FATTURATO_ORDINE','sum'))
    if metric == 'fatturato': agg['METRICA'] = agg['fatturato']
    elif metric == 'kg': agg['METRICA'] = agg['kg']
    else: agg['METRICA'] = np.where(agg['kg']>0, agg['fatturato']/agg['kg'], np.nan)
    agg = _apply_thresholds(agg, thresholds).dropna(subset=['METRICA'])
    out = agg.sort_values('METRICA', ascending=False).head(top_n).copy()
    if out.empty:
        return pd.DataFrame(), "Nessun risultato dopo i filtri/soglie."
    disp = pd.DataFrame({
        grp.upper(): out[grp],
        'Kg Totali': out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", ".")),
        'Fatturato': out['fatturato'].apply(format_euro_robust)
    })
    if metric == 'fatturato':
        disp['Metrica'] = out['fatturato'].apply(format_euro_robust)
    elif metric == 'kg':
        disp['Metrica'] = out['kg'].map(lambda x: f"{x:,.2f} Kg".replace(",", "."))
    else:
        disp['Metrica'] = out['METRICA'].apply(format_euro_robust).str.replace("€ ","€ ")
        disp.rename(columns={'Metrica':'€/Kg medio'}, inplace=True)
    return disp, None

# ---------- MEDIA / MEDIANA ----------
def _compute_avg(df_clienti, df_ordini, entity='clienti', metric='fatturato', anni_list=None, anni_range=None,
                 paese=None, regione=None, how='mean'):
    allowed = _allowed_clients_by_area(df_clienti, anni_list, anni_range, paese, regione)
    if entity == 'clienti':
        # base per cliente
        # fatturato dall'anagrafica, kg dall'ordine
        if metric == 'fatturato':
            dfc = df_clienti[df_clienti['CLIENTE'].isin(allowed)].copy()
            if anni_list: dfc = dfc[dfc['ANNO'].isin(anni_list)]
            if anni_range: dfc = dfc[(dfc['ANNO'] >= anni_range['from']) & (dfc['ANNO'] <= anni_range['to'])]
            agg = dfc.groupby('CLIENTE', as_index=False)['FATTURATO'].sum().rename(columns={'FATTURATO':'val'})
        elif metric == 'kg':
            dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
            if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
            if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
            agg = dfo.groupby('nome_cliente', as_index=False)['KG'].sum().rename(columns={'nome_cliente':'CLIENTE','KG':'val'})
        else:  # euro_kg
            dfo = df_ordini[df_ordini['nome_cliente'].isin(allowed)].copy()
            if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
            if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
            by_cli = dfo.groupby('nome_cliente', as_index=False).agg(Fatt=('FATTURATO_ORDINE','sum'), Kg=('KG','sum'))
            by_cli['val'] = np.where(by_cli['Kg']>0, by_cli['Fatt']/by_cli['Kg'], np.nan)
            agg = by_cli.rename(columns={'nome_cliente':'CLIENTE'})[['CLIENTE','val']].dropna()
        if agg.empty:
            return pd.DataFrame(), "Nessun dato per il calcolo."
        if how == 'median':
            val = float(agg['val'].median())
        else:
            val = float(agg['val'].mean())
        if metric == 'fatturato':
            view = pd.DataFrame({'Valore': [format_euro_robust(val)], 'Metodo':[how]})
        elif metric == 'kg':
            view = pd.DataFrame({'Valore': [f"{val:,.2f} Kg".replace(',', '.')], 'Metodo':[how]})
        else:
            view = pd.DataFrame({'Valore': [format_euro_robust(val).replace("€ ","€ ")], 'Metodo':[how]})
        return view, None

    # articoli / colori
    dfo = df_ordini.copy()
    if allowed: dfo = dfo[dfo['nome_cliente'].isin(allowed)]
    if anni_list: dfo = dfo[dfo['ANNO'].isin(anni_list)]
    if anni_range: dfo = dfo[(dfo['ANNO'] >= anni_range['from']) & (dfo['ANNO'] <= anni_range['to'])]
    grp = 'ARTICOLO' if entity=='articoli' else 'COLORE'
    if metric == 'fatturato':
        agg = dfo.groupby(grp, as_index=False)['FATTURATO_ORDINE'].sum().rename(columns={'FATTURATO_ORDINE':'val'})
    elif metric == 'kg':
        agg = dfo.groupby(grp, as_index=False)['KG'].sum().rename(columns={'KG':'val'})
    else:
        by = dfo.groupby(grp, as_index=False).agg(Fatt=('FATTURATO_ORDINE','sum'), Kg=('KG','sum'))
        by['val'] = np.where(by['Kg']>0, by['Fatt']/by['Kg'], np.nan)
        agg = by[['{}' .format(grp),'val']].dropna()
    if agg.empty:
        return pd.DataFrame(), "Nessun dato per il calcolo."
    val = float(agg['val'].median()) if how=='median' else float(agg['val'].mean())
    if metric == 'fatturato':
        view = pd.DataFrame({'Valore': [format_euro_robust(val)], 'Metodo':[how]})
    elif metric == 'kg':
        view = pd.DataFrame({'Valore': [f"{val:,.2f} Kg".replace(',', '.')], 'Metodo':[how]})
    else:
        view = pd.DataFrame({'Valore': [format_euro_robust(val).replace("€ ","€ ")], 'Metodo':[how]})
    return view, None

# ---------- DISPATCHER PRINCIPALE ----------
def compute_copilot_answer(df_clienti: pd.DataFrame, df_ordini: pd.DataFrame, intent: dict):
    """
    Smista la richiesta verso growth/share/avg/ranking e
    costruisce descrizione metodologica.
    """
    anni_list = intent.get('anni') or None
    anni_range = intent.get('anni_range')
    paese = intent.get('paese')
    regione = intent.get('regione')
    entity = intent.get('entity') or 'clienti'
    metric = intent.get('metric') or ('fatturato' if intent.get('intent')!='growth' else 'fatturato')
    top_n = intent.get('top_n') or 3
    thresholds = intent.get('thresholds') or []

    # GROWTH
    if intent.get('intent') == 'growth':
        basis = 'kg' if metric=='kg' else 'fatturato'
        res, err = _compute_growth_clients(
            df_clienti, df_ordini, basis=basis, anni_list=anni_list, anni_range=anni_range,
            paese=paese, regione=regione, min_start=5000.0 if basis=='fatturato' else 500.0,
            top_n=top_n, prefer='cagr'
        )
        if err:
            return {'table': pd.DataFrame(), 'explain': err, 'assumption': intent}
        explain = f"Crescita calcolata come **CAGR** (e Δ assoluto) su {basis} tra primo e ultimo anno disponibili per ciascun cliente."
        if anni_list or anni_range:
            explain += f" Periodo: **{anni_range or ', '.join(anni_list)}**."
        if paese: explain += f" Area: **{paese}**."
        if regione: explain += f" Regione: **{regione.title()}**."
        explain += " Clienti con almeno 2 anni validi; esclusi valori iniziali troppo bassi."
        return {'table': res, 'explain': explain, 'assumption': intent}

    # SHARE
    if intent.get('intent') == 'share':
        basis = 'kg' if metric=='kg' else 'fatturato'
        res, err = _compute_share(df_clienti, df_ordini, entity=entity, basis=basis,
                                  anni_list=anni_list, anni_range=anni_range,
                                  paese=paese, regione=regione, top_n=top_n)
        if err:
            return {'table': pd.DataFrame(), 'explain': err, 'assumption': intent}
        explain = f"Quota calcolata come (sottoinsieme / totale)×100 sulla base **{basis}**."
        if anni_list or anni_range:
            explain += f" Periodo: **{anni_range or ', '.join(anni_list)}**."
        if paese or regione:
            explain += f" Sottoinsieme: **{paese or regione.title()}**."
        return {'table': res, 'explain': explain, 'assumption': intent}

    # MEDIE / MEDIANE
    if intent.get('intent') in ('mean','median'):
        how = 'median' if intent.get('intent')=='median' else 'mean'
        res, err = _compute_avg(df_clienti, df_ordini, entity=entity, metric=metric,
                                anni_list=anni_list, anni_range=anni_range,
                                paese=paese, regione=regione, how=how)
        if err:
            return {'table': pd.DataFrame(), 'explain': err, 'assumption': intent}
        explain = f"{'Mediana' if how=='median' else 'Media'} calcolata per **{entity}** sulla metrica **{metric}**."
        if anni_list or anni_range:
            explain += f" Periodo: **{anni_range or ', '.join(anni_list)}**."
        if paese: explain += f" Area: **{paese}**."
        if regione: explain += f" Regione: **{regione.title()}**."
        return {'table': res, 'explain': explain, 'assumption': intent}

    # RANKING (default)
    res, err = _compute_ranking(df_clienti, df_ordini, entity=entity, metric=metric,
                                anni_list=anni_list, anni_range=anni_range,
                                paese=paese, regione=regione,
                                thresholds=thresholds, top_n=top_n)
    if err:
        return {'table': pd.DataFrame(), 'explain': err, 'assumption': intent}

    explain = f"Ranking **Top {top_n}** per **{entity}** sulla metrica **{metric}**."
    if anni_list or anni_range:
        explain += f" Periodo: **{anni_range or ', '.join(anni_list)}**."
    if paese: explain += f" Area: **{paese}**."
    if regione: explain += f" Regione: **{regione.title()}**."
    if thresholds:
        txt = "; ".join([f"{t['field']} {t['op']} {t['value']}" for t in thresholds])
        explain += f" Soglie applicate: {txt}."
    return {'table': res, 'explain': explain, 'assumption': intent}

# ---------- PAGINA COPILOT (aggiornata) ----------
def page_copilot(df_clienti: pd.DataFrame, df_ordini: pd.DataFrame, anni_disponibili: list):
    st.title("🤖 Copilot IA — Query Libera su Tutta la Dashboard")
    st.caption("Capisce crescita (CAGR/Δ), quote %, medie/mediane, soglie (> / <), ranking. Lavora su tutta la base dati, indipendente dai filtri globali.")
    q = st.text_input("Scrivi la tua domanda (es. 'Top 3 clienti per crescita % 2019–2023', 'Quota fatturato Italia nel 2024', 'Fatturato medio per cliente 2022', 'Articoli con €/Kg > 8 nel 2023')",
                      key="copilot_free_q")
    add_ai_comment = st.checkbox("Aggiungi commento IA (opzionale)", value=False)
    if st.button("Esegui"):
        if not q.strip():
            st.warning("Inserisci una domanda."); return
        intent = parse_user_query(q)
        res = compute_copilot_answer(df_clienti, df_ordini, intent)

        # Fallback esplicativo se tabella vuota
        st.markdown(f"**Interpretazione automatica:** `{json.dumps(intent, ensure_ascii=False)}`")
        if res["table"].empty:
            st.error("Non sono riuscito a calcolare un risultato con i dati disponibili.")
            st.caption("Suggerimenti: specifica periodo (es. 2021–2024), metrica (fatturato/kg/€/kg), e l'entità (clienti/articoli/colori).")
            return

        st.markdown(res["explain"])
        st.dataframe(res["table"], use_container_width=True, hide_index=True)

        # Commento IA opzionale (riusa la tua call_llm / digest sintetico)
        if add_ai_comment:
            sample = res["table"].head(5).to_dict(orient="records")
            system = ("Sei un assistente analitico per una dashboard B2B. "
                      "Commenta in italiano in max 6 bullet: insight, caveat, next step.")
            user = f"Domanda: {q}\nInterpretazione: {json.dumps(intent, ensure_ascii=False)}\nEstratto tabella: {sample}"
            ai_out = call_llm(system, user)
            st.markdown("**Commento IA:**")
            st.markdown(ai_out)
