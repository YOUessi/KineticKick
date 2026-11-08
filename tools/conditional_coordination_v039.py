"""Conditioned (but NOT robot-executable) temporal prior for 3DSP 3D segment relations.

Predict last 12 frames of 11 pose-derived coordination cosines conditioned on
first 8 frames. Training uses a conditional variational latent model with a
separately supervised deployment prior; all eval predictions use prior mean.
No contact-truth, sport skill-level, or real robot control claim.
"""
from __future__ import annotations
import torch
from torch import nn

class CoordinationCVAE(nn.Module):
    def __init__(self,channels=11,history=8,future=12,hidden=128,latent=12,mean=None,std=None):
        super().__init__()
        self.channels=channels
        self.history=history
        self.future=future
        if mean is None:mean=torch.zeros(channels)
        if std is None:std=torch.ones(channels)
        self.register_buffer("source_mean",torch.as_tensor(mean,dtype=torch.float32).reshape(1,1,channels))
        self.register_buffer("source_std",torch.as_tensor(std,dtype=torch.float32).reshape(1,1,channels))
        self.encoder=nn.Sequential(nn.Linear(history*channels,hidden),nn.ELU(),
                                   nn.Linear(hidden,hidden),nn.ELU())
        self.prior=nn.Linear(hidden,latent*2)
        self.posterior=nn.Sequential(nn.Linear(hidden+future*channels,hidden),
                                     nn.ELU(),nn.Linear(hidden,latent*2))
        self.decoder=nn.Sequential(nn.Linear(hidden+latent,hidden),nn.ELU(),
                                   nn.Linear(hidden,hidden),nn.ELU(),
                                   nn.Linear(hidden,future*channels))
        # Conditioning with a last-observed-pose skip prevents random motion
        # at initialization; learned output changes entire future curve.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def normalized(self,coords):
        return (coords-self.source_mean)/self.source_std

    def posterior_features(self,past,future=None):
        if past.ndim!=3 or past.shape[1:]!=(self.history,self.channels):
            raise ValueError("Expected B x 8 x 11 observed pose relations")
        if not torch.isfinite(past).all():raise ValueError("Nonfinite observed pose relations")
        h=self.encoder(self.normalized(past).reshape(len(past),-1))
        mu_p,lv_p=self.prior(h).chunk(2,dim=-1)
        lv_p=lv_p.clamp(-6,3)
        if future is None:return h,mu_p,lv_p,None,None
        if future.shape!=(len(past),self.future,self.channels) or not torch.isfinite(future).all():
            raise ValueError("Expected finite B x 12 x 11 future")
        mu_q,lv_q=self.posterior(torch.cat([h,self.normalized(future).reshape(len(past),-1)],-1)).chunk(2,dim=-1)
        return h,mu_p,lv_p,mu_q,lv_q.clamp(-6,3)

    def decode(self,context,latent,past):
        delta=self.decoder(torch.cat([context,latent],dim=-1)).reshape(
            len(past),self.future,self.channels)
        last=self.normalized(past)[:,-1:,:]
        normalized_future=last+delta
        return normalized_future*self.source_std+self.source_mean

    def predict(self,past):
        h,mu_p,_,_,_=self.posterior_features(past)
        return self.decode(h,mu_p,past).clamp(-1.,1.)

    def training_objective(self,past,future,kl_weight=.008,posterior_weight=.3):
        h,mu_p,lv_p,mu_q,lv_q=self.posterior_features(past,future)
        eps=torch.randn_like(mu_q)
        z=mu_q+torch.exp(.5*lv_q)*eps
        prediction=self.decode(h,mu_p,past)
        post_prediction=self.decode(h,z,past)
        # Normalize by train-only statistics to balance relation scales.
        target=self.normalized(future)
        pred=self.normalized(prediction)
        post=self.normalized(post_prediction)
        prior_mse=(pred-target).square().mean()
        posterior_mse=(post-target).square().mean()
        kl=.5*(lv_p-lv_q+(lv_q.exp()+(mu_q-mu_p).square())/lv_p.exp()-1.).sum(-1).mean()
        loss=prior_mse+posterior_weight*posterior_mse+kl_weight*kl
        return loss,dict(prior_mse=prior_mse,posterior_mse=posterior_mse,kl=kl)

    @torch.no_grad()
    def sample(self,past,n_samples=4,temperature=1.):
        if not isinstance(n_samples,int) or not 1<=n_samples<=64:raise ValueError("Samples 1..64")
        h,mu,lv,_,_=self.posterior_features(past)
        eps=torch.randn((n_samples,*mu.shape),device=mu.device,dtype=mu.dtype)
        z=mu[None]+temperature*torch.exp(.5*lv)[None]*eps
        return torch.stack([self.decode(h,z[i],past).clamp(-1,1) for i in range(n_samples)],0)

def masked_errors(pred,true,mask):
    """Raw cosine units; no invalid GT treated as zero success."""
    if pred.shape!=true.shape or mask.shape!=true.shape:raise ValueError("Matching shapes required")
    if not bool(mask.any()):raise ValueError("Nonempty measured targets required")
    p=pred[mask];t=true[mask]
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError("Nonfinite valid targets")
    return dict(mse=float(((p-t)**2).mean()),mae=float((p-t).abs().mean()),
                evaluated_scalar_count=int(mask.sum()),mean_persistence_correlation=None)
