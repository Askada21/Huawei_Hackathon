"""Supervised preference objectives. All candidates are real impressions."""
from factor import torch,F

def event_targets(labels):
    return torch.where(labels.sum(1)>0,labels.argmax(1),labels.shape[1])

def event_softmax(logits,labels):
    outside=torch.zeros((len(logits),1),device=logits.device,dtype=logits.dtype)
    return F.cross_entropy(torch.cat((logits,outside),1),event_targets(labels))

def event_pairwise(logits,labels):
    """Winner versus every loser, including the known no-click outcome.

    At zero logits the 2K/(K+1) scale matches ALL candidate-logit gradients
    of (K+1)-class cross entropy for clicked AND no-click events. This
    prevents a trivial learning-rate/regularization scale confound.
    """
    k=logits.shape[1]
    outside=torch.zeros((len(logits),1),device=logits.device,dtype=logits.dtype)
    scores=torch.cat((logits,outside),1);target=event_targets(labels)
    winner=scores.gather(1,target[:,None])
    mask=torch.ones_like(scores,dtype=torch.bool).scatter_(1,target[:,None],False)
    differences=(winner-scores)[mask].reshape(-1,k)
    return F.softplus(-differences).mean()*(2*k/(k+1))

def observed_global_rank(logits,labels,pairs_per_positive=16):
    """Log-probability comparisons across observed labelled rows in a batch.

    Controls come exclusively from clicked=0 training impressions, including
    no-click events, NOT random unobserved catalogue items. Event softmax must
    accompany this shift-invariant ranking objective to anchor no-click mass.
    """
    outside=torch.zeros((len(logits),1),device=logits.device,dtype=logits.dtype)
    logp=logits-torch.logsumexp(torch.cat((logits,outside),1),1,keepdim=True)
    positive=logp[labels>0];negative=logp[labels==0]
    if not len(positive) or not len(negative):return logits.sum()*0
    # Deterministic rotating comparisons: no extra RNG changes the common
    # auxiliary-history sampler or model initialization in controlled ablations.
    index=(torch.arange(len(positive),device=logits.device)[:,None]*37+
           torch.arange(pairs_per_positive,device=logits.device)[None,:]*997)%len(negative)
    return F.softplus(negative[index]-positive[:,None]).mean()
