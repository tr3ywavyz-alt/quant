import os, sys, math, json, random
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ET = ZoneInfo('America/New_York')
CT = ZoneInfo('America/Chicago')
NS_MIN = 60_000_000_000

ICT_DIR = os.path.join(os.path.dirname(__file__), 'ictbot')
SILVER_DIR = os.path.join(os.path.dirname(__file__), 'silver')
sys.path.insert(0, ICT_DIR)

from v106_dynamic_rr_zone_entry import get_liquidity_levels, in_kz
from backtest_entry_modes import gen_sweep_entries_enriched

KZ = [((7,30),(14,30))]
ENTRY_SLIP = 0.25
EXIT_SLIP = 0.25
MNQ_DOLLARS_PER_POINT = 2.0
MNQ_RT_COST = 2.00
DISP_RISK_BUDGET = 300.0
IFVG_RISK_BUDGET = 250.0
GMCL = 4
COOLDOWN_MIN = 2


def normalize_ohlc(df):
    cmap = {c.lower(): c for c in df.columns}
    tcol = None
    for cand in ['datetime_et','datetime','timestamp','date','time']:
        if cand in cmap:
            tcol = cmap[cand]; break
    if tcol is None:
        raise RuntimeError(f'No timestamp column found: {list(df.columns)}')
    def col(name):
        if name.lower() in cmap: return cmap[name.lower()]
        raise RuntimeError(f'Missing {name}')
    out = pd.DataFrame({
        'ts': pd.to_datetime(df[tcol], errors='coerce'),
        'open': pd.to_numeric(df[col('open')], errors='coerce'),
        'high': pd.to_numeric(df[col('high')], errors='coerce'),
        'low': pd.to_numeric(df[col('low')], errors='coerce'),
        'close': pd.to_numeric(df[col('close')], errors='coerce'),
    }).dropna()
    if out['ts'].dt.tz is None:
        try:
            out['ts'] = out['ts'].dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
        except Exception:
            out['ts'] = out['ts'].dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')
            out = out.dropna(subset=['ts'])
    else:
        out['ts'] = out['ts'].dt.tz_convert('America/New_York')
    out = out.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    return out


def to_bar_dicts(df_et, freq):
    x = df_et.copy().set_index('ts')
    x = x.tz_convert('America/Chicago')
    if freq != '1min':
        x = x.resample(freq, label='left', closed='left', origin='start_day').agg(
            open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'))
        x = x.dropna()
    bars=[]
    for ts, r in x.iterrows():
        bars.append({'time_ns': int(ts.tz_convert('UTC').value),
                     'open': float(r.open), 'high': float(r.high), 'low': float(r.low), 'close': float(r.close),
                     'date': ts.date(), 'hour': int(ts.hour), 'minute': int(ts.minute)})
    return bars


def build_dr(bars):
    dr={}
    for i,b in enumerate(bars):
        d=b['date']
        if d not in dr: dr[d]=(i,i+1)
        else: dr[d]=(dr[d][0],i+1)
    return dr


def gen_candidates(b1,b5):
    dr5=build_dr(b5)
    all_dates=sorted(dr5)
    b1_times=np.array([b['time_ns'] for b in b1],dtype=np.int64)
    raw=[]
    for di,d in enumerate(all_dates):
        if d.weekday()>=5: continue
        ds5,de5=dr5[d]
        liq=get_liquidity_levels(b5,dr5,d,all_dates)
        # start b1 one hour before first 5m bar of calendar day
        sess_ns=b5[ds5]['time_ns']
        b1_day_start=int(np.searchsorted(b1_times, sess_ns-3600_000_000_000, side='left'))
        seen=set()
        for cursor in range(ds5+1,de5):
            next_ns=b5[cursor]['time_ns']+5*NS_MIN
            cutoff=int(np.searchsorted(b1_times,next_ns,side='left'))
            ents=gen_sweep_entries_enriched(b5[:cursor+1], b1[b1_day_start:cutoff], ds5, cursor, d, liq, kz=KZ)
            for e in sorted(ents,key=lambda z:(z['ns'],-{'ifvg':2,'disp_fvg':1}.get(z.get('zt'),0))):
                if e['ns'] in seen: continue
                t=datetime.fromtimestamp(e['ns']/1e9,tz=CT)
                if not in_kz(t.hour,t.minute,KZ): continue
                seen.add(e['ns'])
                side=e['side']; zt=e['zt']
                ep=float(e['ep_close'])
                ep += ENTRY_SLIP if side=='bull' else -ENTRY_SLIP
                sp=float(e['sp'])
                risk=abs(ep-sp)
                if risk < 1.0: continue
                rr=1.0 if zt=='ifvg' else 1.1
                tp=ep+risk*rr if side=='bull' else ep-risk*rr
                raw.append({'date':d,'entry_ns':int(e['ns']),'side':side,'zone':zt,'entry':ep,'stop':sp,'target':tp,'risk_pts':risk})
    raw.sort(key=lambda x:x['entry_ns'])
    return raw


def simulate_market(raw, df_et):
    # 1m conservative execution: if stop and target hit in same bar, stop wins.
    ts_ns=np.array([int(t.tz_convert('UTC').value) for t in df_et['ts']],dtype=np.int64)
    op=df_et['open'].to_numpy(float); hi=df_et['high'].to_numpy(float); lo=df_et['low'].to_numpy(float); cl=df_et['close'].to_numpy(float)
    trades=[]; current_day=None; pos_until=0; cool_until=0; consec=0
    for s in raw:
        d=s['date']
        if d!=current_day:
            current_day=d; pos_until=0; cool_until=0; consec=0
        if consec>=GMCL: continue
        en=s['entry_ns']
        if en < pos_until or en < cool_until: continue
        i0=int(np.searchsorted(ts_ns,en,side='left'))
        iend=int(np.searchsorted(ts_ns,en+8*3600_000_000_000_000,side='right'))
        if i0>=len(ts_ns): continue
        side=s['side']; st=s['stop']; tg=s['target']; ep=s['entry']; exit_px=None; exit_ns=None; why='timeout'
        last_idx=min(len(ts_ns)-1,max(i0,iend-1))
        for j in range(i0,min(iend,len(ts_ns))):
            if side=='bull':
                hit_s=lo[j] <= st; hit_t=hi[j] >= tg
            else:
                hit_s=hi[j] >= st; hit_t=lo[j] <= tg
            if hit_s: # includes ambiguous bar -> conservative stop-first
                exit_px=st-EXIT_SLIP if side=='bull' else st+EXIT_SLIP; exit_ns=int(ts_ns[j]); why='loss'; break
            if hit_t:
                exit_px=tg-EXIT_SLIP if side=='bull' else tg+EXIT_SLIP; exit_ns=int(ts_ns[j]); why='win'; break
        if exit_px is None:
            exit_px=float(cl[last_idx]); exit_ns=int(ts_ns[last_idx])
        pts=(exit_px-ep) if side=='bull' else (ep-exit_px)
        r=pts/s['risk_pts']
        trades.append({**s,'exit_ns':exit_ns,'exit':exit_px,'why':why,'r':r})
        pos_until=exit_ns
        cool_until=exit_ns+COOLDOWN_MIN*NS_MIN
        if r<0: consec+=1
        else: consec=0
    return trades


def trade_pnl(t, phase, balance):
    budget=DISP_RISK_BUDGET if t['zone']=='disp_fvg' else IFVG_RISK_BUDGET
    # contract limits: eval 40 micros; funded 20 -> 30 at +1500 -> 40 at +2000
    if phase=='eval': cap=40
    else:
        profit=balance-50000.0
        cap=40 if profit>=2000 else (30 if profit>=1500 else 20)
    per_contract_risk=t['risk_pts']*MNQ_DOLLARS_PER_POINT
    qty=max(0,min(cap,int(math.floor(budget/max(per_contract_risk,1e-9)))))
    if qty<1: return 0.0,0
    points=(t['exit']-t['entry']) if t['side']=='bull' else (t['entry']-t['exit'])
    pnl=points*MNQ_DOLLARS_PER_POINT*qty - MNQ_RT_COST*qty
    return pnl,qty


def run_lifecycle(day_map, dates, start_idx, max_days=20):
    phase='eval'; bal=50000.0; floor=48000.0; eval_best_day=0.0; payouts=0
    for off in range(max_days):
        idx=start_idx+off
        if idx>=len(dates): return 'timeout',off+1
        d=dates[idx]; day_start=bal; day_pnl=0.0
        for t in day_map.get(d,[]):
            if phase=='funded' and day_pnl <= -750: break
            pnl,qty=trade_pnl(t,phase,bal)
            if qty<1: continue
            bal += pnl; day_pnl += pnl
            if bal <= floor:
                return 'blow',off+1
            if phase=='funded' and day_pnl <= -1000:
                return 'blow',off+1
        if phase=='eval':
            eval_best_day=max(eval_best_day,day_pnl)
            # EOD trail for next day
            floor=max(floor,bal-2000.0)
            total=bal-50000.0
            consistency=(eval_best_day <= 0.40*total) if total>0 else False
            if total>=3000.0 and consistency:
                phase='funded'; bal=50000.0; floor=48000.0; payouts=0
        else:
            # EOD trail, locking at 50,100 once balance >=52,100
            if bal>=52100: floor=50100.0
            else: floor=max(floor,bal-2000.0)
            if payouts==0 and bal>=53100:
                bal-=1000.0; payouts=1
            elif payouts==1 and bal>=53100:
                bal-=1000.0; payouts=2
                return 'success',off+1
    return 'timeout',max_days


def historical_lifecycle(trades):
    day_map=defaultdict(list)
    for t in trades: day_map[t['date']].append(t)
    dates=sorted(day_map)
    # include weekdays with no trades between first/last date
    full=[]; d=dates[0]
    while d<=dates[-1]:
        if d.weekday()<5: full.append(d)
        d += timedelta(days=1)
    out=[]
    for i in range(0,max(0,len(full)-20+1)):
        status,days=run_lifecycle(day_map,full,i,20)
        out.append((status,days,full[i]))
    return out, day_map, full


def bootstrap_lifecycle(day_map, full_dates, n=50000, block=5, seed=42):
    rng=random.Random(seed)
    valid_starts=list(range(0,max(1,len(full_dates)-block)))
    counts=defaultdict(int); daycounts=[]
    for _ in range(n):
        synth=[]
        while len(synth)<20:
            s=rng.choice(valid_starts)
            synth.extend(full_dates[s:s+block])
        synth=synth[:20]
        # remap synthetic dates to artificial sequential dates but preserve each sampled day's trades
        fake_map={}
        fake_dates=[]
        base=datetime(2030,1,1).date()
        for k,srcd in enumerate(synth):
            fd=base+timedelta(days=k)
            fake_dates.append(fd)
            fake_map[fd]=[{**t,'date':fd} for t in day_map.get(srcd,[])]
        st,dy=run_lifecycle(fake_map,fake_dates,0,20)
        counts[st]+=1; daycounts.append(dy)
    return counts,daycounts


def stats(trades):
    if not trades: return {}
    wins=[t for t in trades if t['r']>0]; losses=[t for t in trades if t['r']<0]
    gw=sum(t['r'] for t in wins); gl=-sum(t['r'] for t in losses)
    eq=0; peak=0; mdd=0
    for t in trades:
        eq+=t['r']; peak=max(peak,eq); mdd=max(mdd,peak-eq)
    return {'n':len(trades),'wr':len(wins)/len(trades),'pf':gw/gl if gl else 999,'avg_r':sum(t['r'] for t in trades)/len(trades),'maxdd_r':mdd}


def main():
    path=os.path.join(SILVER_DIR,'nq_1m.parquet')
    print('DATA PATH',path,os.path.exists(path),os.path.getsize(path) if os.path.exists(path) else None)
    df=normalize_ohlc(pd.read_parquet(path))
    print('COVERAGE',df.ts.iloc[0],df.ts.iloc[-1],'ROWS',len(df))
    b1=to_bar_dicts(df,'1min'); b5=to_bar_dicts(df,'5min')
    print('BARS',len(b1),len(b5))
    raw=gen_candidates(b1,b5)
    print('RAW SIGNALS',len(raw))
    trades=simulate_market(raw,df)
    s=stats(trades); print('TRADE_STATS',json.dumps(s,indent=2))
    byy={}
    for y in sorted(set(t['date'].year for t in trades)):
        byy[y]=stats([t for t in trades if t['date'].year==y])
    print('YEARLY',json.dumps(byy,indent=2))
    hold=stats([t for t in trades if t['date']>=datetime(2024,1,1).date()])
    print('HOLDOUT_2024_PLUS',json.dumps(hold,indent=2))
    hist,day_map,full=historical_lifecycle(trades)
    hc=defaultdict(int)
    for st,dy,d in hist: hc[st]+=1
    n=len(hist)
    print('ROLLING20',json.dumps({k:v/n if n else 0 for k,v in hc.items()},indent=2),'N',n)
    bc,days=bootstrap_lifecycle(day_map,full,n=50000,block=5,seed=20260819)
    print('BOOTSTRAP50K',json.dumps({k:v/50000 for k,v in bc.items()},indent=2))
    print('MEDIAN_DAYS',float(np.median(days)))
    # Write compact artifacts
    pd.DataFrame(trades).to_csv('v106_trades.csv',index=False)
    summary={'coverage':[str(df.ts.iloc[0]),str(df.ts.iloc[-1])],'rows':len(df),'trade_stats':s,'yearly':byy,'holdout':hold,
             'rolling20':{k:v/n if n else 0 for k,v in hc.items()},'rolling_n':n,
             'bootstrap':{k:v/50000 for k,v in bc.items()},'median_days':float(np.median(days))}
    with open('v106_summary.json','w') as f: json.dump(summary,f,indent=2)

if __name__=='__main__': main()
