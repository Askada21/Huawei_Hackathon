"""Compact history ablations with past-only spectral initialization.

No LightGBM import: export calibration_features.npz in this Torch process, then
fit/apply combiners in a separate process. Existing factors/artifacts untouched.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from factor import Factors, torch, nn, F
from common import (ROOT, RUNS, load, split, cutoff, dimensions, priors,
                    interactions, record, roc_auc_score)


class HistoryFactors(Factors):
    def __init__(self, nu, ni, dim, stats, mode='baseline', half_life=24.,
                 separate_history=False, train_history=False, gates=False,
                 gate_mode='reliability',context_rank=0,context_mode='bucket-day',temporal_gates=False,time_history=False,cold_history=False,profile_shrinkage=0.,engagement_history=False,nonlinear_affinity=False):
        if temporal_gates and not gates:
            raise ValueError('temporal-gates requires gates')
        if profile_shrinkage<0 or (profile_shrinkage and not cold_history):
            raise ValueError('positive profile-shrinkage requires a cold-history path')
        super().__init__(nu, ni, dim, stats)
        self.mode = mode
        self.half_life = half_life
        self.separate_history = separate_history
        self.use_gates = gates
        self.gate_mode = gate_mode
        self.cold_history=cold_history
        self.profile_shrinkage=profile_shrinkage
        if engagement_history:
            self.engagement_weights = nn.Parameter(torch.zeros(6))
        if cold_history:self.cold_hw=nn.Parameter(self.hw.detach().clone())
        if gate_mode not in ['global','reliability']:raise ValueError(gate_mode)
        self.attention_temperature = nn.Parameter(torch.tensor(1.0))
        if separate_history:
            self.context = nn.Embedding(ni, dim, padding_idx=0)
            self.context.weight.requires_grad_(train_history)
        # Neutral initialization gives exactly the baseline reliability weights.
        self.gate = nn.Linear(2, 2)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.register_buffer('user_exposure', torch.tensor(stats[0], dtype=torch.float32))
        self.context_rank=context_rank
        self.context_mode=context_mode
        self.temporal_gates=temporal_gates
        self.time_history=time_history
        if time_history:
            self.history_bucket_logits=nn.Parameter(torch.zeros(4,4))
        if temporal_gates:
            self.temporal_gate=nn.Embedding(4,2,_weight=torch.zeros(4,2))
        if context_rank:
            contexts=4 if context_mode=='bucket' else 8
            self.context_left=nn.Parameter(torch.randn(contexts,dim,context_rank)*.05)
            self.context_right=nn.Parameter(torch.zeros(contexts,context_rank,dim))
        if nonlinear_affinity:
            self.affinity_hidden = nn.Linear(dim, 16)
            self.affinity_out = nn.Linear(16, 1, bias=False)
            nn.init.zeros_(self.affinity_out.weight)

    def contextualize(self,pref,contexts):
        if not self.context_rank:return pref
        left=self.context_left[contexts];right=self.context_right[contexts]
        if pref.ndim==2:
            low=torch.einsum('bd,bdr->br',pref,left)
            residual=torch.einsum('br,brd->bd',low,right)
        else:
            low=torch.einsum('bcd,bdr->bcr',pref,left)
            residual=torch.einsum('bcr,brd->bcd',low,right)
        return pref+residual/(self.context_rank**.5)

    def score(self, u, i, h, b, day, timestamp=None, history_time=None,user_keep=None):
        mask = h.gt(0)
        n = mask.sum(1, keepdim=True).clamp_min(1).float()
        uv, iv = self.u(u), self.i(i)
        if user_keep is not None:
            if user_keep.shape!=(len(u),1):raise ValueError('user_keep must have shape (events, 1)')
            uv=uv*user_keep
        profile_weight=None
        if self.profile_shrinkage:
            exposure=self.user_exposure[u]
            profile_weight=(exposure/(exposure+self.profile_shrinkage))[:,None]
            if user_keep is not None:profile_weight=profile_weight*user_keep
            uv=uv*profile_weight
        hvec = self.context(h) if self.separate_history else self.i(h)
        if self.mode == 'baseline':
            if self.time_history:
                if history_time is None:
                    raise ValueError('time-history requires history timestamps')
                # UTC buckets; only each event's own strictly earlier history is read.
                hb=(history_time//21600)%4
                weights=2*self.history_bucket_logits[b[:,None],hb].sigmoid()*mask
                scale=weights.square().sum(1,keepdim=True).clamp_min(1e-8).sqrt()
                hv=(hvec*weights[...,None]).sum(1)/scale
            else:
                hv = (hvec * mask[..., None]).sum(1) / n.sqrt()
        else:
            if timestamp is None or history_time is None:
                raise ValueError('Recency/candidate modes require prediction-time history timestamps')
            # Padded times are excluded. Clamp only avoids multiplying large pad ages.
            age = ((timestamp[:, None] - history_time).float() / 3600).clamp_min(0)
            decay = -np.log(2.) * age / self.half_life
            if self.mode == 'recency':
                logits = decay.masked_fill(~mask, -1e4)
                weights = logits.softmax(-1) * mask
                hv = (hvec * weights[..., None]).sum(1) * n.sqrt()
            elif self.mode == 'candidate':
                similarity = torch.einsum('bcd,bhd->bch', F.normalize(iv, dim=-1),
                                          F.normalize(hvec, dim=-1))
                temp = self.attention_temperature.clamp(0, 5)
                logits = (similarity * temp + decay[:, None, :]).masked_fill(~mask[:, None, :], -1e4)
                weights = logits.softmax(-1) * mask[:, None, :]
                hv = torch.einsum('bch,bhd->bcd', weights, hvec) * n.sqrt()[:, :, None]
            else:
                raise ValueError(self.mode)
        if self.use_gates:
            exposure=self.user_exposure[u]
            if user_keep is not None:exposure=exposure*user_keep[:,0]
            info = torch.stack((torch.log1p(mask.sum(1).float()) / 3,
                                torch.log1p(exposure) / 5), -1)
            if self.gate_mode == 'global':info = torch.zeros_like(info)
            gate_logits=self.gate(info)
            if self.temporal_gates:gate_logits=gate_logits+self.temporal_gate(b)
            gates = 2 * gate_logits.sigmoid()
            uv = uv * gates[:, :1]
            hv = hv * (gates[:, 1:, None] if hv.ndim == 3 else gates[:, 1:])
        history_weight=self.hw
        if self.cold_history:
            missing=self.user_exposure[u].eq(0)
            if user_keep is not None:missing=missing|~user_keep[:,0].bool()
            if profile_weight is not None:
                history_weight=profile_weight*self.hw+(1-profile_weight)*self.cold_hw
            else:history_weight=torch.where(missing,self.cold_hw,self.hw)[:,None]
        if hv.ndim == 3:
            if self.cold_history:history_weight=history_weight[:,:,None]
            pref=self.contextualize(uv[:,None,:]+history_weight*hv,b if self.context_mode=='bucket' else b+day*4)
            products = pref * iv
        else:
            pref=self.contextualize(uv+history_weight*hv,b if self.context_mode=='bucket' else b+day*4)
            products = pref[:,None,:] * iv
        affinity = products.sum(-1)
        if hasattr(self, 'affinity_out'):
            affinity = affinity + self.affinity_out(F.gelu(self.affinity_hidden(products))).squeeze(-1)
        user_bias=self.ub(u);user_time=self.ut(u).gather(1,b[:,None])
        if user_keep is not None:
            user_bias=user_bias*user_keep;user_time=user_time*user_keep
        if profile_weight is not None:
            user_bias=user_bias*profile_weight;user_time=user_time*profile_weight
        z = affinity + user_bias + self.ib(i).squeeze(-1) + self.global_bias
        z = z + self.tb(b + day * 4)
        z = z + user_time
        z = z + self.it(i).gather(2, b[:, None, None].expand(-1, i.shape[1], 1)).squeeze(-1)
        if hasattr(self, 'engagement_weights'):
            if timestamp is None or history_time is None:
                raise ValueError('History engagement requires timestamps')
            age = (timestamp[:, None] - history_time).float().clamp_min(0) / 3600
            latest = age.masked_fill(~mask, 24 * 30).amin(dim=1).clamp_max(24 * 30)
            features = torch.stack([
                mask.sum(1) / 20,
                torch.log1p(latest) / 7,
                (mask & (age <= 6)).sum(1) / 20,
                (mask & (age <= 24)).sum(1) / 20,
                (mask & (age <= 72)).sum(1) / 20,
                (~mask.any(1)).float(),
            ], dim=1)
            z = z + (features @ self.engagement_weights)[:, None]
        return z + self.repeat * (i[:, :, None] == h[:, None, :]).sum(-1)


def batch(a, indices):
    return [torch.tensor(np.asarray(a[k][indices]), dtype=torch.long) for k in
            ['user', 'item', 'history', 'bucket', 'day', 'time', 'htime']]


def selection_mask(a, events, end_day=None):
    """Restrict checkpoint selection without dropping exported validation rows."""
    if end_day is None:
        return np.ones(len(events), dtype=bool)
    return np.asarray(a['time'][events] < cutoff(end_day))


def select_basis(su, si, dim, mode='leading'):
    """Select paired columns without outcomes, excluding the leading component.

    Effective participation is scale invariant. Taking the smaller user/item
    participation rejects components localized on either side of the graph.
    Stable component index resolves exact ties deterministically.
    """
    if su.shape[1] != si.shape[1] or not 0 < dim < su.shape[1]:
        raise ValueError('Need aligned user/item components and 0 < dim < rank')
    def participation(a):
        a=np.asarray(a,dtype='float64')
        sq=np.square(a)
        return np.square(sq.sum(0))/np.maximum(np.square(sq).sum(0),1e-300)
    pu,pi=participation(su),participation(si)
    distributed=np.minimum(pu,pi)
    candidates=np.arange(1,su.shape[1])
    if mode=='leading':indices=candidates[:dim]
    elif mode=='distributed':
        indices=candidates[np.lexsort((candidates,-distributed[candidates]))[:dim]]
    else:raise ValueError(mode)
    metadata={'mode':mode,'selected_indices':indices.tolist(),
              'effective_users':pu.tolist(),'effective_items':pi.tolist(),
              'min_effective_participation':distributed.tolist(),
              'selection_uses_labels':False,'leading_component_excluded':True}
    return su[:,indices],si[:,indices],metadata


@torch.no_grad()
def predict(model, a, events, batch_size=2048):
    model.eval(); zs=[]; ps=[]
    for s in range(0, len(events), batch_size):
        z = model.score(*batch(a, events[s:s+batch_size]))
        p = torch.softmax(torch.cat((z, torch.zeros((len(z), 1))), 1), 1)[:, :10]
        zs.append(z.numpy()); ps.append(p.numpy())
    return np.concatenate(ps), np.concatenate(zs)


def construct(args, stats, nu, ni):
    return HistoryFactors(nu, ni, args.dim, stats, args.mode, args.half_life,
                          args.separate_history, args.train_history, args.gates,
                          getattr(args,'gate_mode','reliability'),getattr(args,'context_rank',0),getattr(args,'context_mode','bucket-day'),getattr(args,'temporal_gates',False),getattr(args,'time_history',False),getattr(args,'user_dropout',0)>0,getattr(args,'profile_shrinkage',0.),getattr(args,'engagement_history',False),getattr(args,'nonlinear_affinity',False))


def export(path):
    """48 columns exactly matching the old clean combiner's feature contract."""
    from tree_features import features
    args = argparse.Namespace(**json.loads((path/'config.json').read_text()))
    def digest(p):
        with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    model_digest=digest(path/'model.pt')
    a = load(); nu, ni = dimensions(); fit, events = split(a, args.day)
    if args.full:
        fit = np.arange(len(a['time'])); aout = load('evaluation'); events = np.arange(len(aout['time']))
    else:
        aout = a
    stats = priors(a, fit, nu, ni); model = construct(args, stats, nu, ni)
    model.load_state_dict(torch.load(path/'model.pt', map_location='cpu', weights_only=True))
    p, z = predict(model, aout, events)
    x, names = features(aout, events, stats); cols=[x]
    def add(name, value):
        value = np.asarray(value, dtype='float32')
        if value.ndim == 1: value=np.repeat(value[:, None], 10, 1)
        cols.append(value.reshape(-1, 1)); names.append(name)
    add('factor_probability', p); add('factor_logit', z)
    add('factor_logit_minus_mean', z-z.mean(1, keepdims=True))
    add('factor_logit_minus_max', z-z.max(1, keepdims=True))
    add('factor_rank', (z[:, :, None] < z[:, None, :]).sum(-1))
    add('event_p', p.sum(1)); add('event_max', p.max(1))
    add('event_entropy', -(p*np.log(p.clip(1e-9))).sum(1)); add('event_score_std', z.std(1))
    wi=model.i.weight.detach().numpy(); wi=wi/np.maximum(np.linalg.norm(wi,axis=-1,keepdims=True),1e-8)
    geometries=[]
    for s in range(0,len(events),2048):
        e=events[s:s+2048];h=aout['history'][e];ii=aout['item'][e];n=(h>0).sum(1)
        sim=np.einsum('bcd,bhd->bch',wi[ii],wi[h])
        mean=sim.sum(-1)/n[:,None].clip(1)
        maximum=np.where(h[:,None,:]>0,sim,-1).max(-1)
        last=sim[np.arange(len(e))[:,None],np.arange(10)[None,:],np.maximum(n-1,0)[:,None]]
        geometries.append(np.stack((mean,maximum,last),-1))
    geo=np.concatenate(geometries)
    for k,name in enumerate(['history_cosine_mean','history_cosine_max','history_cosine_last']):add(name,geo[:,:,k])
    x=np.concatenate(cols,1)
    if digest(path/'model.pt')!=model_digest:
        raise RuntimeError('Checkpoint changed during export; wait for training completion and export again')
    np.savez(path/'calibration_features.npz', x=x, names=names, y=aout['label'][events].ravel(),
             user=aout['user'][events],events=events,p=p)
    np.save(path/('evaluation_pred.npy' if args.full else 'valid_pred.npy'),p)
    (path/'feature_provenance.json').write_text(json.dumps({'model_sha256':model_digest,
        'config_sha256':digest(path/'config.json'),'events':len(events),'feature_names':names,
        'full':args.full,'day':args.day},indent=2))
    print('Exported',path,x.shape,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--day',type=int,default=26);ap.add_argument('--dim',type=int,default=32)
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--lr',type=float,default=.0024);ap.add_argument('--reg',type=float,default=1.)
    ap.add_argument('--aux',type=float,default=.5);ap.add_argument('--anchor',type=float,default=0.)
    ap.add_argument('--mode',choices=['baseline','recency','candidate'],default='baseline')
    ap.add_argument('--half-life',type=float,default=24.)
    ap.add_argument('--separate-history',action='store_true');ap.add_argument('--train-history',action='store_true')
    ap.add_argument('--gates',action='store_true');ap.add_argument('--freeze-item',action='store_true')
    ap.add_argument('--gate-mode',choices=['global','reliability'],default='reliability')
    ap.add_argument('--basis-mode',choices=['leading','distributed'],default='leading')
    ap.add_argument('--init-normalization',choices=['unit','soft'],default='unit')
    ap.add_argument('--supervised-loss',choices=['softmax','pairwise','hybrid','global-hybrid'],default='softmax')
    ap.add_argument('--context-reg',type=float,default=1.)
    ap.add_argument('--context-anchor',type=float,default=1.)
    ap.add_argument('--aux-controls',choices=['unobserved','observed'],default='unobserved')
    ap.add_argument('--context-rank',type=int,default=0)
    ap.add_argument('--context-mode',choices=['bucket-day','bucket'],default='bucket-day')
    ap.add_argument('--temporal-gates',action='store_true')
    ap.add_argument('--time-history',action='store_true')
    ap.add_argument('--contextual-aux',action='store_true')
    ap.add_argument('--spectral-degree-reg',type=float,default=0.)
    ap.add_argument('--spectral-negative-weight', type=float, default=0.)
    ap.add_argument('--spectral-adaptive-negatives', action='store_true')
    ap.add_argument('--spectral-half-life-days', type=float, default=0.)
    ap.add_argument('--engagement-history', action='store_true')
    ap.add_argument('--nonlinear-affinity', action='store_true')
    ap.add_argument('--aux-history', type=int, default=0,
                    help='Preceding fitting clicks used in the auxiliary ranking task')
    ap.add_argument('--user-dropout',type=float,default=0.)
    ap.add_argument('--profile-shrinkage',type=float,default=0.)
    ap.add_argument('--tag',default='v3_history');ap.add_argument('--full',action='store_true')
    ap.add_argument('--selection-end-day',type=int)
    ap.add_argument('--export',type=Path);ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.export:export(args.export);return
    if args.half_life<=0:raise ValueError('half-life must be positive')
    if args.spectral_degree_reg<0:raise ValueError('spectral-degree-reg must be nonnegative')
    if not np.isfinite(args.spectral_negative_weight) or args.spectral_negative_weight < 0:
        raise ValueError('spectral-negative-weight must be finite and nonnegative')
    if not np.isfinite(args.spectral_half_life_days) or args.spectral_half_life_days < 0:
        raise ValueError('spectral-half-life-days must be finite and nonnegative')
    if not 0 <= args.aux_history <= 20:
        raise ValueError('aux-history must be between 0 and 20')
    if not 0<=args.user_dropout<1:raise ValueError('user-dropout must be in [0,1)')
    if args.profile_shrinkage<0 or (args.profile_shrinkage and not args.user_dropout):
        raise ValueError('positive profile-shrinkage requires user-dropout')
    if args.time_history and args.mode!='baseline':raise ValueError('time-history requires baseline pooling')
    if args.contextual_aux and (args.context_rank<=0 or args.context_mode!='bucket' or args.aux_controls!='observed'):
        raise ValueError('contextual-aux requires bucket contextual factors and observed controls')
    if args.train_history and not args.separate_history:raise ValueError('train-history requires separate-history')
    np.random.seed(args.seed);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    a=load();nu,ni=dimensions();fit,v=split(a,args.day)
    if args.full:fit=np.arange(len(a['time']));v=np.array([],dtype=int)
    select=selection_mask(a,v,args.selection_end_day)
    if len(v) and not select.any():raise ValueError('No validation events inside selection period')
    if len(v) and np.unique(a['label'][v[select]]).size<2:raise ValueError('Selection period must contain both target classes')
    out=RUNS/f'{args.tag}_d{args.dim}_day{args.day}_s{args.seed}{"_full" if args.full else ""}'
    out.mkdir(exist_ok=False)
    config=vars(args).copy();config['export']=None
    (out/'config.json').write_text(json.dumps(config,indent=2))
    np.save(out/'valid_events.npy',v)
    stats=priors(a,fit,nu,ni);model=construct(args,stats,nu,ni)
    # Exactly the previous selected spectral initialization and scaling.
    from spectral import embeddings
    # Preserve the exact legacy cache key for the neutral default (0, not 0.0).
    degree_reg=0 if args.spectral_degree_reg==0 else args.spectral_degree_reg
    su,si=embeddings(a,fit,nu,ni,rank=max(64,args.dim+1),n_iter=25,degree_reg=degree_reg,
                     negative_weight=args.spectral_negative_weight, half_life_days=args.spectral_half_life_days,
                     adaptive_negatives=args.spectral_adaptive_negatives)
    su,si,basis=select_basis(su,si,args.dim,args.basis_mode)
    (out/'basis.json').write_text(json.dumps(basis,indent=2))
    print('Spectral basis selection:',json.dumps({'mode':basis['mode'],'indices':basis['selected_indices'],
          'min_participation':[basis['min_effective_participation'][j] for j in basis['selected_indices']]}),flush=True)
    with torch.no_grad():
        for emb,arr in [(model.u,su),(model.i,si)]:
            norm=np.linalg.norm(arr,axis=1,keepdims=True)
            if args.init_normalization=='soft':
                active=norm[:,0]>1e-8
                tau=np.median(norm[active]**2)
                arr=arr/np.sqrt(norm**2+tau)
                arr=arr*(1.5/np.sqrt(np.mean(np.sum(arr[active]**2,axis=1))))
            else:arr=arr/np.maximum(norm,1e-8)*1.5
            emb.weight.copy_(torch.tensor(arr))
        if args.separate_history:model.context.weight.copy_(model.i.weight)
    if args.freeze_item:model.i.weight.requires_grad_(False)
    anchor_u=model.u.weight.detach().clone();anchor_i=model.i.weight.detach().clone()
    context_start=model.context.weight.detach().clone() if args.separate_history else None
    clicks=interactions(a,fit,nu,ni)
    probs=np.bincount(clicks[:,2],minlength=ni).astype('float64')**.75;probs[0]=0;probs/=probs.sum()
    if args.aux_controls=='observed':
        from preference_sampling_v5 import observed_negative_pool,observed_bucket_negative_pool
        pool=observed_bucket_negative_pool if args.contextual_aux else observed_negative_pool
        negative_items,negative_counts,negative_start=pool(a,fit,nu)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    best=-1.;stale=0;start=time.time()
    drop_rng=torch.Generator().manual_seed(args.seed+14000)
    for ep in range(args.epochs):
        model.train();order=rng.permutation(fit);losses=[]
        for s in range(0,len(order),2048):
            ix=order[s:s+2048];u,i,h,b,d,t,ht=batch(a,ix)
            keep=(torch.rand((len(ix),1),generator=drop_rng)>=args.user_dropout) if args.user_dropout else None
            z=model.score(u,i,h,b,d,t,ht,user_keep=keep);y=torch.tensor(np.asarray(a['label'][ix]),dtype=torch.float32)
            target=torch.where(y.sum(1)>0,y.argmax(1),10)
            if args.supervised_loss in ['pairwise','hybrid']:
                from objectives_v5 import event_pairwise
                loss=event_pairwise(z,y)
                if args.supervised_loss=='hybrid':
                    loss=.5*loss+.5*F.cross_entropy(torch.cat((z,torch.zeros((len(z),1))),1),target)
            else:
                loss=F.cross_entropy(torch.cat((z,torch.zeros((len(z),1))),1),target)
                if args.supervised_loss=='global-hybrid':
                    from objectives_v5 import observed_global_rank
                    loss=.5*loss+.5*observed_global_rank(z,y)
            loss=loss+args.reg*sum(x.square().mean() for x in [model.u(u),model.i(i),model.ub(u),model.ib(i)])
            if args.context_rank:
                residual=model.context_left@model.context_right
                loss=loss+args.reg*residual.square().mean()
            if args.temporal_gates:loss=loss+args.reg*model.temporal_gate.weight.square().mean()
            if args.time_history:loss=loss+args.reg*model.history_bucket_logits.square().mean()
            if args.separate_history and args.train_history and h.gt(0).any():
                ids=h[h.gt(0)];context=model.context(ids)
                loss=loss+args.context_reg*context.square().mean()+args.context_anchor*(context-context_start[ids]).square().mean()
            if args.anchor:loss=loss+args.anchor*((model.u(u)-anchor_u[u]).square().mean()+(model.i(i)-anchor_i[i]).square().mean())
            if args.aux:
                click_indices = rng.integers(len(clicks),size=len(ix))
                c=clicks[click_indices]
                if args.aux_controls=='observed':
                    keys=c[:,0]*4+(c[:,1]//21600)%4 if args.contextual_aux else c[:,0]
                    eligible=negative_counts[keys]>0;c=c[eligible];keys=keys[eligible]
                    click_indices = click_indices[eligible]
                cu=torch.tensor(c[:,0],dtype=torch.long);ci=torch.tensor(c[:,2],dtype=torch.long)
                if len(c):
                    if args.aux_controls=='observed':
                        positions=negative_start[keys,None]+(rng.random((len(c),4))*negative_counts[keys,None]).astype('int64')
                        neg=torch.tensor(negative_items[positions],dtype=torch.long)
                    else:neg=torch.tensor(rng.choice(ni,(len(c),4),p=probs),dtype=torch.long)
                    av=model.u(cu);pv=model.i(ci);nv=model.i(neg)
                    if args.aux_history:
                        previous = click_indices[:, None] - np.arange(1, args.aux_history + 1)
                        earlier = clicks[previous.clip(min=0)]
                        # interactions sorts by user and time. Exclude same-minute clicks.
                        seen = ((previous >= 0) & (earlier[:, :, 0] == c[:, None, 0])
                                & (earlier[:, :, 1] < c[:, None, 1]))
                        mask = torch.from_numpy(seen)
                        vectors = model.i(torch.tensor(earlier[:, :, 2], dtype=torch.long))
                        history = (vectors * mask[..., None]).sum(1)
                        av = av + .5 * history / mask.sum(1, keepdim=True).clamp_min(1).sqrt()
                    if args.contextual_aux:
                        cb=torch.tensor((c[:,1]//21600)%4,dtype=torch.long)
                        av=model.contextualize(av,cb)
                    losses_aux=F.softplus(-(av[:,None,:]*(pv[:,None,:]-nv)).sum(-1))
                    usable=neg.ne(ci[:,None]) if args.aux_controls=='observed' else torch.ones_like(neg,dtype=torch.bool)
                    if usable.any():loss=loss+args.aux*losses_aux[usable].mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        result={'epoch':ep+1,'loss':float(np.mean(losses)),'seconds':time.time()-start}
        if len(v):
            p,z=predict(model,a,v);auc=roc_auc_score(a['label'][v].ravel(),p.ravel());result['auc']=auc
            selection_auc=roc_auc_score(a['label'][v[select]].ravel(),p[select].ravel())
            result['selection_auc']=selection_auc;result['selection_events']=int(select.sum())
            if selection_auc>best:
                best=selection_auc;stale=0;torch.save(model.state_dict(),out/'model.pt')
                np.save(out/'valid_pred.npy',p);np.save(out/'valid_logits.npy',z)
                (out/'best.json').write_text(json.dumps(result,indent=2))
            else:stale+=1
        else:torch.save(model.state_dict(),out/'model.pt')
        print(json.dumps(result),flush=True)
        with open(out/'epochs.jsonl','a') as f:f.write(json.dumps(result)+'\n')
        if stale>=4:break
    if len(v):
        saved=np.load(out/'valid_pred.npy')
        record(out.name,args.day,a['label'][v],saved,config=config,seconds=time.time()-start,artifact=str(out),
               selection_auc=float(roc_auc_score(a['label'][v[select]].ravel(),saved[select].ravel())),selection_events=int(select.sum()))
    else:export(out)


if __name__=='__main__':main()
