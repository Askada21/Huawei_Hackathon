"""Controlled preservation of the spectral starting geometry; old model intact."""
import argparse,json,time
import numpy as np
from factor import Factors,batch,predict,torch,F
from common import *
from spectral import embeddings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--day',type=int,default=26);ap.add_argument('--mode',choices=['frozen','anchored','staged'],required=True)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--anchor',type=float,default=10)
    ap.add_argument('--lr',type=float,default=.0024);ap.add_argument('--full',action='store_true')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--tag',default='stable')
    ap.add_argument('--selection-end-day',type=int)
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    a=load();nu,ni=dimensions();fit,v=split(a,args.day)
    if args.full:fit=np.arange(len(a['time']));v=np.array([],dtype=int)
    selection=np.ones(len(v),dtype=bool) if args.selection_end_day is None else a['time'][v]<cutoff(args.selection_end_day)
    if len(v):assert selection.any()
    name=f'{args.tag}_{args.mode}_d32_day{args.day}_s{args.seed}'+('_full' if args.full else '')
    out=RUNS/name;out.mkdir(exist_ok=False)
    config=dict(vars(args),dim=32,no_history=False,cosine=False,objective='softmax',family='stable_v3',reg=1,aux=.5)
    (out/'config.json').write_text(json.dumps(config,indent=2));np.save(out/'valid_events.npy',v)
    stats=priors(a,fit,nu,ni);model=Factors(nu,ni,32,stats)
    su,si=embeddings(a,fit,nu,ni,rank=64,n_iter=25,degree_reg=0)
    with torch.no_grad():
        for emb,arr in [(model.u,su),(model.i,si)]:
            arr=arr[:,1:33];arr=arr/np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-8)*1.5
            emb.weight.copy_(torch.tensor(arr))
    anchor_u=model.u.weight.detach().clone();anchor_i=model.i.weight.detach().clone()
    clicks=interactions(a,fit,nu,ni);rng=np.random.default_rng(args.seed)
    probs=np.bincount(clicks[:,2],minlength=ni).astype('float64')**.75;probs[0]=0;probs/=probs.sum()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=-1;stale=0;start=time.time()
    for ep in range(args.epochs):
        model.i.weight.requires_grad_(args.mode!='frozen' and not(args.mode=='staged' and ep<2))
        model.train();order=rng.permutation(fit);losses=[]
        for s in range(0,len(order),2048):
            ix=order[s:s+2048];u,i,h,b,d=batch(a,ix)
            z=model.score(u,i,h,b,d);y=torch.tensor(np.asarray(a['label'][ix]),dtype=torch.float32)
            target=torch.where(y.sum(1)>0,y.argmax(1),10)
            loss=F.cross_entropy(torch.cat([z,torch.zeros((len(z),1))],1),target)
            loss=loss+sum(x.square().mean() for x in [model.u(u),model.i(i),model.ub(u),model.ib(i)])
            if args.mode in ['anchored','staged']:
                loss=loss+args.anchor*((model.u(u)-anchor_u[u]).square().mean()+(model.i(i)-anchor_i[i]).square().mean())
            c=clicks[rng.integers(len(clicks),size=len(ix))]
            cu=torch.tensor(c[:,0],dtype=torch.long);ci=torch.tensor(c[:,2],dtype=torch.long)
            neg=torch.tensor(rng.choice(ni,(len(c),4),p=probs),dtype=torch.long)
            av=model.u(cu);pv=model.i(ci);nv=model.i(neg)
            loss=loss+.5*F.softplus(-(av[:,None,:]*(pv[:,None,:]-nv)).sum(-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        info={'epoch':ep+1,'loss':float(np.mean(losses)),'seconds':time.time()-start}
        if len(v):
            p,z=predict(model,a,v);info['auc']=float(roc_auc_score(a['label'][v].ravel(),p.ravel()))
            info['selection_auc']=float(roc_auc_score(a['label'][v[selection]].ravel(),p[selection].ravel()))
            if info['selection_auc']>best:
                best=info['selection_auc'];stale=0;torch.save(model.state_dict(),out/'model.pt')
                np.save(out/'valid_pred.npy',p);np.save(out/'valid_logits.npy',z)
                (out/'best.json').write_text(json.dumps(info))
            else:stale+=1
        print(json.dumps(info),flush=True)
        with open(out/'epochs.jsonl','a') as f:f.write(json.dumps(info)+'\n')
        if stale>=4:break
    if len(v):record(name,args.day,a['label'][v],np.load(out/'valid_pred.npy'),config=config,seconds=time.time()-start,artifact=str(out))
    else:
        torch.save(model.state_dict(),out/'model.pt');p,z=predict(model,load('evaluation'),np.arange(52939))
        np.save(out/'evaluation_pred.npy',p);np.save(out/'evaluation_logits.npy',z)

if __name__=='__main__':main()
