import os, sys, math, json, random
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ET=ZoneInfo('America/New_York'); CT=ZoneInfo('America/Chicago')
ICT_DIR=os.path.join(os.path.dirname(__file__),'ictbot'); SILVER_DIR=os.path.join(os.path.dirname(__file__),'silver')
sys.path.insert(0,ICT_DIR)
from v106_dynamic_rr_zone_entry import get_liquidity_levels, detect_sweep_at

ENTRY_SLIP=.25; EXIT_SLIP=.25; COST_RT=2.00; MNQ_PV=2.0
DISP_BUDGET=300.; IFVG_BUDGET=250.; GMCL=4; COOLDOWN_MIN=2
KZ_START=(8,0); KZ_END=(14,30)
SOURCE_MAX_RISK_PTS=1000/(20*3)  # exact source $1000 risk gate at 3 NQ


def load_data():
    p=os.path.join(SILVER_DIR,'nq_1m.parquet')
    df=pd.read_parquet(p)
    cmap={c.lower():c for c in df.columns}
    tc=cmap.get('datetime_et') or cmap.get('datetime') or cmap.get('timestamp')
    if not tc: raise RuntimeError(str(df.columns))
    x=pd.DataFrame({'ts':pd.to_datetime(df[tc],errors='coerce'),'open':pd.to_numeric(df[cmap['open']],errors='coerce'),
                    'high':pd.to_numeric(df[cmap['high']],errors='coerce'),'low':pd.to_numeric(df[cmap['low']],errors='coerce'),
                    'close':pd.to_numeric(df[cmap['close']],errors='coerce')}).dropna()
    if x.ts.dt.tz is None:
        try: x['ts']=x.ts.dt.tz_localize('America/New_York',ambiguous='infer',nonexistent='shift_forward')
        except Exception:
            x['ts']=x.ts.dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='shift_forward'); x=x.dropna()
    else: x['ts']=x.ts.dt.tz_convert('America/New_York')
    x=x.sort_values('ts').drop_duplicates('ts').set_index('ts')
    m1=x.tz_convert('America/Chicago')
    m5=m1.resample('5min',label='left',closed='left',origin='start_day').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last')).dropna()
    return m1,m5


def bars_from_m5(m5):
    out=[]
    for ts,r in m5.iterrows():
        out.append({'time_ns':int(ts.tz_convert('UTC').value),'open':float(r.open),'high':float(r.high),'low':float(r.low),'close':float(r.close),
                    'date':ts.date(),'hour':int(ts.hour),'minute':int(ts.minute)})
    return out

def build_dr(bars):
    d={}
    for i,b in enumerate(bars):
        if b['date'] not in d: d[b['date']]=(i,i+1)
        else:d[b['date']]=(d[b['date']][0],i+1)
    return d

def in_kz_ts(ts):
    m=ts.hour*60+ts.minute; return 8*60 <= m < 14*60+30


def _nm_from_leg(leg,n):
    vals=[]
    arr=leg[['open','high','low','close']].to_numpy(float); idx=list(leg.index)
    if n==1:
        for i,row in enumerate(arr): vals.append({'open':row[0],'high':row[1],'low':row[2],'close':row[3],'ts':idx[i]})
        return vals
    for i in range(0,len(arr)-n+1,n):
        g=arr[i:i+n]
        if len(g)<n: break
        vals.append({'open':g[0,0],'high':float(g[:,1].max()),'low':float(g[:,2].min()),'close':g[-1,3],'ts':idx[i]})
    return vals

def _find_fvgs(arr,direction,min_gap=.5,maxw=12.):
    fs=[]
    for k in range(len(arr)-2):
        p,n=arr[k],arr[k+2]
        if direction=='bull':
            gap=p['low']-n['high']; top=p['low']; bot=n['high']
            if gap<min_gap: continue
        else:
            gap=n['low']-p['high']; top=n['low']; bot=p['high']
            if gap<min_gap: continue
        if top-bot>maxw: continue
        bad=False
        for z in arr[k+3:]:
            if direction=='bull' and z['close']<bot: bad=True; break
            if direction=='bear' and z['close']>top: bad=True; break
        if not bad: fs.append({'top':float(top),'bot':float(bot)})
    return fs


def ifvg_candidate(m1,b5,sw_bar,disp_idx,side,sw_lvl):
    sw_ts=pd.Timestamp(b5[sw_bar]['time_ns'],unit='ns',tz='UTC').tz_convert('America/Chicago')
    dp_ts=pd.Timestamp(b5[disp_idx]['time_ns'],unit='ns',tz='UTC').tz_convert('America/Chicago')
    leg=m1[(m1.index>=sw_ts-pd.Timedelta(minutes=15))&(m1.index<dp_ts)]
    if len(leg)<3:return None
    best=None
    for tf in [5,4,3,2,1]:
        fs=_find_fvgs(_nm_from_leg(leg,tf),side)
        if len(fs)==1: best=fs[0]; break
    if best is None:
        for tf in [5,4,3,2,1]:
            fs=_find_fvgs(_nm_from_leg(leg,tf),side)
            if fs:
                best=max(fs,key=lambda f:f['top']) if side=='bull' else min(fs,key=lambda f:f['bot']); break
    if best is None:return None
    inv=best['top'] if side=='bull' else best['bot']
    st=dp_ts+pd.Timedelta(minutes=5); en=st+pd.Timedelta(minutes=20)
    scan=m1[(m1.index>=st)&(m1.index<en)]
    legbars=b5[sw_bar:disp_idx+1]
    if not legbars:return None
    leg_ext=min(b['low'] for b in legbars) if side=='bull' else max(b['high'] for b in legbars)
    sp=min(sw_lvl,leg_ext)-1 if side=='bull' else max(sw_lvl,leg_ext)+1
    for ts,r in scan.iterrows():
        if not in_kz_ts(ts):continue
        rng=r.high-r.low
        if rng<=0:continue
        br=abs(r.close-r.open)/rng
        ok=(side=='bull' and r.close>r.open and r.close>inv+.5 and br>=.35) or (side=='bear' and r.close<r.open and r.close<inv-.5 and br>=.35)
        if not ok:continue
        ep=float(r.close)+(ENTRY_SLIP if side=='bull' else -ENTRY_SLIP); risk=abs(ep-sp)
        if risk<1 or risk>SOURCE_MAX_RISK_PTS:return None
        tp=ep+risk if side=='bull' else ep-risk
        return {'date':ts.date(),'entry_ts':ts+pd.Timedelta(minutes=1),'side':side,'zone':'ifvg','entry':ep,'stop':float(sp),'target':float(tp),'risk_pts':float(risk)}
    return None


def disp_candidates(m1,m5,b5,sw_bar,disp_idx,side,sw_lvl):
    out=[]; seen=set()
    disp_ts=pd.Timestamp(b5[disp_idx]['time_ns'],unit='ns',tz='UTC').tz_convert('America/Chicago'); disp_end=disp_ts+pd.Timedelta(minutes=5)
    day=b5[disp_idx]['date']
    for k in range(sw_bar,disp_idx+2):
        if k<1 or k+1>=len(b5) or b5[k+1]['date']!=day:continue
        p,c,n=b5[k-1],b5[k],b5[k+1]
        if side=='bull':
            gap=n['low']-p['high']
            if gap<=0:continue
            top,bot=float(n['low']),float(p['high'])
        else:
            gap=p['low']-n['high']
            if gap<=0:continue
            top,bot=float(p['low']),float(n['high'])
        zkey=(round(top,2),round(bot,2))
        if zkey in seen:continue
        seen.add(zkey)
        confirm=pd.Timestamp(n['time_ns'],unit='ns',tz='UTC').tz_convert('America/Chicago')+pd.Timedelta(minutes=5)
        start=max(confirm,disp_end)
        if not in_kz_ts(start):continue
        # first completed 5m invalidation after confirmation
        m5after=m5[(m5.index>=confirm)&(m5.index.normalize()==confirm.normalize())]
        invalid=None
        for t5,r5 in m5after.iterrows():
            close_t=t5+pd.Timedelta(minutes=5)
            if side=='bull' and r5.close<bot: invalid=close_t; break
            if side=='bear' and r5.close>top: invalid=close_t; break
        end=invalid if invalid is not None else pd.Timestamp(confirm.date(),tz=CT)+pd.Timedelta(hours=14,minutes=30)
        scan=m1[(m1.index>=start)&(m1.index<end)]
        for ts,r in scan.iterrows():
            if not in_kz_ts(ts):break
            touch=(side=='bull' and r.low<=top and r.close>=bot) or (side=='bear' and r.high>=bot and r.close<=top)
            if not touch:continue
            # last completed 5m bar at the start of this 1m candle
            prior=m5[(m5.index+pd.Timedelta(minutes=5)<=ts)&(m5.index.date==ts.date())]
            if len(prior):
                # structure from bars after displacement, completed before touch
                pr=prior[prior.index>=disp_end]
            else: pr=prior
            if side=='bull':
                pl=float(pr.low.min()) if len(pr) else float(r.low)
                sp=min(float(sw_lvl),pl,float(r.low))-1
            else:
                ph=float(pr.high.max()) if len(pr) else float(r.high)
                sp=max(float(sw_lvl),ph,float(r.high))+1
            ep=float(r.close)+(ENTRY_SLIP if side=='bull' else -ENTRY_SLIP); risk=abs(ep-sp)
            if risk<1 or risk>SOURCE_MAX_RISK_PTS:break
            rr=1.1; tp=ep+risk*rr if side=='bull' else ep-risk*rr
            out.append({'date':ts.date(),'entry_ts':ts+pd.Timedelta(minutes=1),'side':side,'zone':'disp_fvg','entry':ep,'stop':float(sp),'target':float(tp),'risk_pts':float(risk)})
            break
    return out


def gen_signals(m1,m5,b5):
    dr=build_dr(b5); dates=sorted(dr); signals=[]; global_seen=set()
    for di,d in enumerate(dates):
        if d.weekday()>=5:continue
        ds,de=dr[d]; liq=get_liquidity_levels(b5,dr,d,dates)
        # only completed premarket levels: first trade signal at/after 08:00 CT
        ks=[i for i in range(ds,de) if (b5[i]['hour']*60+b5[i]['minute'])>=480 and (b5[i]['hour']*60+b5[i]['minute'])<870]
        if not ks:continue
        shs=[]; sls=[]; pending_sh=None; pending_sl=None; setups={}
        for i in ks:
            if pending_sh is not None: shs.append(pending_sh); pending_sh=None
            if pending_sl is not None: sls.append(pending_sl); pending_sl=None
            if i>ds+1:
                if b5[i-1]['high']>b5[i-2]['high'] and b5[i-1]['high']>b5[i]['high']: pending_sh=(b5[i-1]['high'],'ses_sh')
                if b5[i-1]['low']<b5[i-2]['low'] and b5[i-1]['low']<b5[i]['low']: pending_sl=(b5[i-1]['low'],'ses_sl')
            live=liq+shs[-6:]+sls[-6:]
            sd,sl,sb=detect_sweep_at(b5,i,live,lookback=12)
            if sd is None:continue
            disp=None
            for j in range(sb,min(sb+6,i+1)):
                br=b5[j]['high']-b5[j]['low']
                if br<=0:continue
                body=abs(b5[j]['close']-b5[j]['open'])/br
                if body>=.35 and ((sd=='bull' and b5[j]['close']>b5[j]['open']) or (sd=='bear' and b5[j]['close']<b5[j]['open'])):
                    disp=j;break
            if disp is None:continue
            key=(sb,disp,sd,round(float(sl),2)); setups[key]=(sb,disp,sd,float(sl))
        for sb,disp,sd,sl in setups.values():
            cands=[]
            iv=ifvg_candidate(m1,b5,sb,disp,sd,sl)
            if iv:cands.append(iv)
            cands+=disp_candidates(m1,m5,b5,sb,disp,sd,sl)
            cands.sort(key=lambda x:(x['entry_ts'],0 if x['zone']=='ifvg' else 1))
            for c in cands[:2]:
                ns=int(c['entry_ts'].tz_convert('UTC').value)
                if ns in global_seen:continue
                global_seen.add(ns);c['entry_ns']=ns;signals.append(c)
        if (di+1)%100==0:print('SIGNAL_PROGRESS',di+1,'/',len(dates),'signals',len(signals),flush=True)
    signals.sort(key=lambda x:x['entry_ns']);return signals


def simulate(signals,m1):
    idx=m1.index; ns=np.array([int(t.tz_convert('UTC').value) for t in idx],dtype=np.int64)
    hi=m1.high.to_numpy(float);lo=m1.low.to_numpy(float);cl=m1.close.to_numpy(float)
    trades=[];day=None;busy=0;cool=0;closs=0
    for s in signals:
        if s['date']!=day:day=s['date'];busy=0;cool=0;closs=0
        if closs>=GMCL or s['entry_ns']<busy or s['entry_ns']<cool:continue
        i0=int(np.searchsorted(ns,s['entry_ns'],'left'));iend=int(np.searchsorted(ns,s['entry_ns']+8*3600_000_000_000_000,'right'))
        if i0>=len(ns):continue
        ex=None;ens=None;why='timeout'
        for j in range(i0,min(iend,len(ns))):
            hs=(lo[j]<=s['stop']) if s['side']=='bull' else (hi[j]>=s['stop'])
            ht=(hi[j]>=s['target']) if s['side']=='bull' else (lo[j]<=s['target'])
            if hs:
                ex=s['stop']-EXIT_SLIP if s['side']=='bull' else s['stop']+EXIT_SLIP;ens=int(ns[j]);why='loss';break
            if ht:
                ex=s['target']-EXIT_SLIP if s['side']=='bull' else s['target']+EXIT_SLIP;ens=int(ns[j]);why='win';break
        if ex is None:
            j=max(i0,min(len(ns)-1,iend-1));ex=float(cl[j]);ens=int(ns[j])
        pts=(ex-s['entry']) if s['side']=='bull' else (s['entry']-ex);r=pts/s['risk_pts']
        t={**s,'exit':float(ex),'exit_ns':ens,'why':why,'r':float(r)};trades.append(t)
        busy=ens;cool=ens+COOLDOWN_MIN*60_000_000_000
        closs=closs+1 if r<0 else 0
    return trades


def tstats(ts):
    if not ts:return {}
    w=[t for t in ts if t['r']>0];l=[t for t in ts if t['r']<0];gw=sum(t['r'] for t in w);gl=-sum(t['r'] for t in l)
    eq=peak=dd=0
    for t in ts:eq+=t['r'];peak=max(peak,eq);dd=max(dd,peak-eq)
    return {'n':len(ts),'wr':len(w)/len(ts),'pf':gw/gl if gl else 999,'avg_r':sum(t['r'] for t in ts)/len(ts),'maxdd_r':dd}

def pnl_trade(t,phase,bal):
    budget=DISP_BUDGET if t['zone']=='disp_fvg' else IFVG_BUDGET
    cap=40 if phase=='eval' else (40 if bal-50000>=2000 else (30 if bal-50000>=1500 else 20))
    qty=min(cap,int(math.floor(budget/max(t['risk_pts']*MNQ_PV,1e-9))))
    if qty<1:return 0,0
    pts=(t['exit']-t['entry']) if t['side']=='bull' else (t['entry']-t['exit'])
    return pts*MNQ_PV*qty-COST_RT*qty,qty

def lifecycle(daymap,dates,start,maxd=20):
    phase='eval';bal=50000.;floor=48000.;best=0.;pays=0
    for off in range(maxd):
        if start+off>=len(dates):return 'timeout',off+1
        d=dates[start+off];dp=0.
        for t in daymap.get(d,[]):
            if phase=='funded' and dp<=-750:break
            p,q=pnl_trade(t,phase,bal)
            if q<1:continue
            bal+=p;dp+=p
            if bal<=floor:return 'blow',off+1
            if phase=='funded' and dp<=-1000:return 'blow',off+1
        if phase=='eval':
            best=max(best,dp);floor=max(floor,bal-2000);tot=bal-50000
            if tot>=3000 and best<=.4*tot:phase='funded';bal=50000;floor=48000;pays=0
        else:
            floor=50100 if bal>=52100 else max(floor,bal-2000)
            if pays==0 and bal>=53100:bal-=1000;pays=1
            elif pays==1 and bal>=53100:return 'success',off+1
    return 'timeout',maxd

def lifecycle_tests(trades):
    dm=defaultdict(list)
    for t in trades:dm[t['date']].append(t)
    ds=sorted(dm);full=[];d=ds[0]
    while d<=ds[-1]:
        if d.weekday()<5:full.append(d)
        d+=timedelta(days=1)
    hc=defaultdict(int)
    for i in range(max(0,len(full)-19)):
        st,_=lifecycle(dm,full,i);hc[st]+=1
    hn=sum(hc.values())
    rng=random.Random(20260819);bc=defaultdict(int);bd=[]
    starts=list(range(max(1,len(full)-5)))
    for _ in range(50000):
        src=[]
        while len(src)<20:
            s=rng.choice(starts);src+=full[s:s+5]
        src=src[:20];base=datetime(2030,1,1).date();fd=[];fm={}
        for k,sday in enumerate(src):
            x=base+timedelta(days=k);fd.append(x);fm[x]=[{**t,'date':x} for t in dm.get(sday,[])]
        st,dy=lifecycle(fm,fd,0);bc[st]+=1;bd.append(dy)
    return {k:v/hn for k,v in hc.items()},hn,{k:v/50000 for k,v in bc.items()},float(np.median(bd))

def main():
    m1,m5=load_data();print('COVERAGE',m1.index[0],m1.index[-1],'ROWS',len(m1),flush=True)
    b5=bars_from_m5(m5);print('5M',len(b5),flush=True)
    sig=gen_signals(m1,m5,b5);print('SIGNALS',len(sig),flush=True)
    tr=simulate(sig,m1);print('STATS',json.dumps(tstats(tr),indent=2),flush=True)
    yr={y:tstats([t for t in tr if t['date'].year==y]) for y in sorted(set(t['date'].year for t in tr))};print('YEARLY',json.dumps(yr,indent=2),flush=True)
    hold=tstats([t for t in tr if t['date']>=datetime(2024,1,1).date()]);print('HOLDOUT',json.dumps(hold,indent=2),flush=True)
    rh,n,bs,med=lifecycle_tests(tr);print('ROLLING20',json.dumps(rh),'N',n,flush=True);print('BOOTSTRAP50K',json.dumps(bs),'MEDIAN',med,flush=True)
    pd.DataFrame(tr).to_csv('v106_stream_trades.csv',index=False)
    out={'coverage':[str(m1.index[0]),str(m1.index[-1])],'rows':len(m1),'stats':tstats(tr),'yearly':yr,'holdout':hold,'rolling20':rh,'rolling_n':n,'bootstrap':bs,'median_days':med}
    open('v106_stream_summary.json','w').write(json.dumps(out,indent=2))
if __name__=='__main__':main()
