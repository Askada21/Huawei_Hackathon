"""Stream official CSVs into compact, checked event arrays. No labels inferred."""
import csv
import json
import time
from pathlib import Path
from itertools import groupby
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'arena_data'

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'metadata.json').exists():
        print('Prepared cache already exists; leaving it unchanged.'); return
    users, items = {}, {}
    def encode(mapping, value):
        if value not in mapping: mapping[value] = len(mapping)+1
        return mapping[value]
    for name, count in [('train',686824), ('evaluation',52939)]:
        specs = {'user':('int32',(count,)), 'item':('int32',(count,10)),
                 'time':('int64',(count,)), 'bucket':('int8',(count,)),
                 'day':('int8',(count,)), 'history':('int32',(count,20)),
                 'htime':('int64',(count,20)), 'label':('int8',(count,10))}
        a={k:np.lib.format.open_memmap(OUT/f'{name}_{k}.npy', mode='w+',dtype=d,shape=s)
           for k,(d,s) in specs.items()}
        rowids=[]; previous=-1; n=0; seen=set(); start=time.time()
        with open(ROOT/f'{name}.csv',newline='') as f:
            for event,g in groupby(csv.DictReader(f),key=lambda r:r['event_id']):
                rows=list(g); first=rows[0]; t=int(first['timestamp'])
                assert len(rows)==10 and n<count and event not in seen
                seen.add(event)
                assert t>=previous, 'CSV must be chronological'
                previous=t
                h=json.loads(first['history_item_ids']); ht=json.loads(first['history_timestamps'])
                assert len(h)==len(ht)<=20 and ht==sorted(ht) and all(x<t for x in ht)
                for r in rows:
                    assert all(r[k]==first[k] for k in ['user_id','timestamp','time_bucket','day_type','history_item_ids','history_timestamps'])
                a['user'][n]=encode(users,first['user_id'])
                a['time'][n]=t; a['bucket'][n]=int(first['time_bucket']);a['day'][n]=int(first['day_type'])
                a['item'][n]=[encode(items,r['item_id']) for r in rows]
                a['history'][n]=0; a['htime'][n]=0
                a['history'][n,:len(h)]=[encode(items,x) for x in h]
                a['htime'][n,:len(h)]=ht
                a['label'][n]=[int(r['clicked']) for r in rows] if name=='train' else -1
                if name=='train': assert a['label'][n].min()>=0 and a['label'][n].sum()<=1
                else: rowids.extend(r['rowid'] for r in rows)
                n+=1
                if n%100000==0: print(name,n,'events',round(time.time()-start,1),'sec',flush=True)
        assert n==count
        for x in a.values(): x.flush()
        if rowids:
            assert len(set(rowids))==len(rowids)
            np.save(OUT/'evaluation_rowids.npy',np.asarray(rowids,dtype='U40'))
        print(name,'complete',n,flush=True)
    (OUT/'metadata.json').write_text(json.dumps({'users':users,'items':items,'source_sizes':{n:(ROOT/f'{n}.csv').stat().st_size for n in ['train','evaluation']} }))
    print('Users',len(users),'items',len(items),flush=True)

if __name__=='__main__': main()
