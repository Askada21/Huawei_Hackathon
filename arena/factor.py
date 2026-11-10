"""Joint history-aware factor model; explicit 11-way event likelihood.

Warm-up clicks are positive observations with unobserved sampled controls, never
treated as impression labels. Supervised stage uses all ten observed candidates.
"""
import argparse,sys,time,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.techarena_deps'))
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.special import expit
from common import *

torch.set_num_threads(4)

class Factors(nn.Module):
    def __init__(self,nu,ni,d,stats,history=True,cosine=False):
        super().__init__();self.history=history;self.cosine=cosine
        self.u=nn.Embedding(nu,d,padding_idx=0);self.i=nn.Embedding(ni,d,padding_idx=0)
        self.ub=nn.Embedding(nu,1);self.ib=nn.Embedding(ni,1)
        self.tb=nn.Embedding(8,1);self.ut=nn.Embedding(nu,4);self.it=nn.Embedding(ni,4)
        self.global_bias=nn.Parameter(torch.tensor(-1.8));self.hw=nn.Parameter(torch.tensor(0.5))
        self.repeat=nn.Parameter(torch.tensor(-.3))
        self.scale=nn.Parameter(torch.tensor(5.0))
        for emb in [self.u,self.i]: nn.init.normal_(emb.weight,std=.08)
        for emb in [self.ut,self.it]: nn.init.zeros_(emb.weight)
        un,uc,inn,ic,g=stats
        with torch.no_grad():
            self.ub.weight[:,0].copy_(torch.tensor(np.log((uc+10*g)/(un-uc+10*(1-g)))-np.log(g/(1-g)),dtype=torch.float32))
            self.ib.weight[:,0].copy_(torch.tensor(np.log((ic+20*g)/(inn-ic+20*(1-g)))-np.log(g/(1-g)),dtype=torch.float32))
            self.tb.weight.zero_();self.u.weight[0].zero_();self.i.weight[0].zero_()

    def score(self,u,i,h=None,b=None,day=None):
        uv=self.u(u)
        if self.history and h is not None:
            hv=self.i(h).sum(1)/(h.gt(0).sum(1,keepdim=True).clamp_min(1).float().sqrt())
            uv=uv+self.hw*hv
        iv=self.i(i)
        if self.cosine:
            affinity=(F.normalize(uv,dim=-1)[:,None,:]*F.normalize(iv,dim=-1)).sum(-1)*self.scale
        else: affinity=(uv[:,None,:]*iv).sum(-1)
        z=affinity+self.ub(u)+self.ib(i).squeeze(-1)+self.global_bias
        if b is not None:
            z=z+self.tb(b+day*4)
            z=z+torch.gather(self.ut(u),1,b[:,None])+torch.gather(self.it(i),2,b[:,None,None].expand(-1,i.shape[1],1)).squeeze(-1)
        if h is not None:
            z=z+self.repeat*(i[:,:,None]==h[:,None,:]).sum(-1)
        return z

def batch(a,idx):
    return [torch.tensor(np.asarray(a[k][idx]),dtype=torch.long) for k in ['user','item','history','bucket','day']]

@torch.no_grad()
def predict(model,a,events,objective='softmax'):
    model.eval();ps=[];zs=[]
    for start in range(0,len(events),4096):
        z=model.score(*batch(a,events[start:start+4096]));zs.append(z.numpy())
        if objective=='softmax': p=torch.softmax(torch.cat([z,torch.zeros((len(z),1))],1),1)[:,:10]
        else:p=z.sigmoid()
        ps.append(p.numpy())
    return np.concatenate(ps),np.concatenate(zs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--day',type=int,default=26);ap.add_argument('--dim',type=int,default=32)
    ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--pretrain',type=int,default=8)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr',type=float,default=.008)
    ap.add_argument('--reg',type=float,default=.001);ap.add_argument('--no-history',action='store_true')
    ap.add_argument('--objective',choices=['softmax','binary'],default='softmax')
    ap.add_argument('--tag',default='factor');ap.add_argument('--full',action='store_true')
    ap.add_argument('--svd-init',action='store_true')
    ap.add_argument('--cosine',action='store_true');ap.add_argument('--anchor',type=float,default=0)
    ap.add_argument('--als-init',action='store_true')
    ap.add_argument('--svd-iters',type=int,default=5)
    ap.add_argument('--aux',type=float,default=0)
    ap.add_argument('--degree-reg',type=float,default=0)
    args=ap.parse_args();np.random.seed(args.seed);torch.manual_seed(args.seed)
    a=load();nu,ni=dimensions();fit,v=split(a,args.day)
    if args.full: fit=np.arange(len(a['time']));v=np.array([],dtype=int)
    name=f'{args.tag}_d{args.dim}_day{args.day}_s{args.seed}'+('_full' if args.full else '')
    out=RUNS/name;out.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    np.save(out/'valid_events.npy',v)
    stats=priors(a,fit,nu,ni);model=Factors(nu,ni,args.dim,stats,not args.no_history,args.cosine)
    if args.svd_init:
        from spectral import embeddings
        su,si=embeddings(a,fit,nu,ni,rank=max(64,args.dim+1),n_iter=args.svd_iters,degree_reg=args.degree_reg)
        with torch.no_grad():
            for emb,arr in [(model.u,su),(model.i,si)]:
                arr=arr[:,1:args.dim+1]
                arr=arr/np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-8)*1.5
                emb.weight.copy_(torch.tensor(arr))
    if args.als_init:
        from als import embeddings
        su,si=embeddings(a,fit,nu,ni,rank=args.dim)
        with torch.no_grad():
            for emb,arr in [(model.u,su),(model.i,si)]:
                arr=arr/np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-8)*1.5
                emb.weight.copy_(torch.tensor(arr))
    anchor_u=model.u.weight.detach().clone();anchor_i=model.i.weight.detach().clone()
    clicks=interactions(a,fit,nu,ni)
    # Positive-only histories augment preference learning. Deduplication avoids
    # multiplying a click by how many subsequent events repeat it.
    rng=np.random.default_rng(args.seed)
    probs=np.bincount(clicks[:,2],minlength=ni).astype('float64')**.75;probs[0]=0;probs/=probs.sum()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    start=time.time()
    for ep in range(args.pretrain):
        order=rng.permutation(len(clicks));losses=[]
        for s in range(0,len(order),4096):
            c=clicks[order[s:s+4096]];u=torch.tensor(c[:,0],dtype=torch.long)
            pos=torch.tensor(c[:,2],dtype=torch.long)
            neg=torch.tensor(rng.choice(ni,(len(c),8),p=probs),dtype=torch.long)
            uv=model.u(u);pv=model.i(pos);nv=model.i(neg)
            diff=(uv[:,None,:]*(pv[:,None,:]-nv)).sum(-1)
            loss=F.softplus(-diff).mean()+args.reg*(uv.square().mean()+pv.square().mean()+nv.square().mean())
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        p,z=predict(model,a,v,args.objective) if len(v) else (None,None)
        auc=roc_auc_score(a['label'][v].ravel(),p.ravel()) if len(v) else None
        print('pretrain',ep+1,'loss',np.mean(losses),'auc',auc,'seconds',round(time.time()-start),flush=True)
    # Lower step size for observed-label fitting.
    opt=torch.optim.Adam(model.parameters(),lr=args.lr*.3)
    best=-1;stale=0
    for ep in range(args.epochs):
        model.train();order=rng.permutation(fit);losses=[]
        for s in range(0,len(order),2048):
            ix=order[s:s+2048];u,i,h,b,d=batch(a,ix)
            z=model.score(u,i,h,b,d);y=torch.tensor(np.asarray(a['label'][ix]),dtype=torch.float32)
            if args.objective=='softmax':
                target=torch.where(y.sum(1)>0,y.argmax(1),10)
                loss=F.cross_entropy(torch.cat([z,torch.zeros((len(z),1))],1),target)
            else:loss=F.binary_cross_entropy_with_logits(z,y)
            reg=sum(x.square().mean() for x in [model.u(u),model.i(i),model.ub(u),model.ib(i)])
            loss=loss+args.reg*reg
            if args.anchor:
                loss=loss+args.anchor*((model.u(u)-anchor_u[u]).square().mean()+(model.i(i)-anchor_i[i]).square().mean())
            if args.aux:
                c=clicks[rng.integers(len(clicks),size=len(ix))]
                cu=torch.tensor(c[:,0],dtype=torch.long);ci=torch.tensor(c[:,2],dtype=torch.long)
                neg=torch.tensor(rng.choice(ni,(len(c),4),p=probs),dtype=torch.long)
                av=model.u(cu);pv=model.i(ci);nv=model.i(neg)
                loss=loss+args.aux*F.softplus(-(av[:,None,:]*(pv[:,None,:]-nv)).sum(-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        if len(v):
            p,z=predict(model,a,v,args.objective);auc=roc_auc_score(a['label'][v].ravel(),p.ravel())
            print('epoch',ep+1,'loss',np.mean(losses),'auc',auc,'seconds',round(time.time()-start),flush=True)
            with open(out/'epochs.jsonl','a') as f:f.write(json.dumps({'epoch':ep+1,'auc':auc,'seconds':time.time()-start})+'\n')
            if auc>best:
                best=auc;stale=0;torch.save(model.state_dict(),out/'model.pt');np.save(out/'valid_pred.npy',p);np.save(out/'valid_logits.npy',z)
                (out/'best.json').write_text(json.dumps({'epoch':ep+1,'auc':auc}))
            else:stale+=1
            if stale>=5:break
        else:
            print('epoch',ep+1,'loss',np.mean(losses),'seconds',round(time.time()-start),flush=True)
    if len(v):
        record(name,args.day,a['label'][v],np.load(out/'valid_pred.npy'),config=vars(args),seconds=time.time()-start,artifact=str(out))
    else:
        torch.save(model.state_dict(),out/'model.pt')
        p,z=predict(model,load('evaluation'),np.arange(52939),args.objective)
        np.save(out/'evaluation_pred.npy',p);np.save(out/'evaluation_logits.npy',z)

if __name__=='__main__':main()
