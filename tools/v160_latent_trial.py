#!/usr/bin/env python3
"""Paired, source-separated CPU study of causal 200 ms whole-body reference forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/v160_latent_trial.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clip_splits(definition):
    groups = {name: list(definition[name]) for name in ('train', 'validation', 'test')}
    ids = [c for clips in groups.values() for c in clips]
    if len(set(ids)) != len(ids) or any(not group for group in groups.values()):
        raise ValueError('Each clip must belong to exactly one nonempty partition')
    return groups


@dataclass
class Motion:
    name: str
    side: str
    joints: np.ndarray
    root_p: np.ndarray
    root_r: Rotation
    sha256: str


def load_motion(path, name, source, schema, expected):
    digest = sha(path)
    if digest != expected:
        raise ValueError(f'Changed input motion {path}')
    with np.load(path, allow_pickle=False) as raw:
        q = np.asarray(raw['joint_pos'], dtype=np.float64)
        if np.asarray(raw['fps']).size != 1 or float(np.asarray(raw['fps']).item()) != 50:
            raise ValueError(f'Expected a 50 Hz clip: {path}')
        if q.ndim != 2 or q.shape[1] != 29 or len(q) < 18 or not np.isfinite(q).all():
            raise ValueError(f'Invalid 29-joint clip {path}')
        pos = np.asarray(raw['body_pos_w'], dtype=np.float64)
        orient = np.asarray(raw['body_quat_w'], dtype=np.float64)
        if pos.shape != (len(q),30,3) or orient.shape != (len(q),30,4):
            raise ValueError('Expected 30 ordered body transforms')
        if source == 'cmu':
            if list(raw['joint_names']) != schema['joint_names'] or list(raw['body_names']) != schema['body_names']:
                raise ValueError('CMU G1 reference order differs from declared runtime')
            side = 'unknown'
        else:
            side = str(raw['kick_leg'].item())
            if side not in ('left','right') or not name.endswith('_'+side):
                raise ValueError('Inconsistent source kicking-foot annotation')
        quat = orient[:,schema['body_names'].index('pelvis')]
        root_position = pos[:,schema['body_names'].index('pelvis')]
        if not np.isfinite(root_position).all() or not np.isfinite(quat).all() or not np.allclose(np.linalg.norm(quat,axis=-1),1.,atol=2e-4):
            raise ValueError('Invalid pelvis trajectory')
        return Motion(name, side, q, root_position, Rotation.from_quat(quat[:,[1,2,3,0]]), digest)


@dataclass
class Windows:
    context: np.ndarray
    target: np.ndarray
    anchors: np.ndarray


def make_windows(motion, history, horizon, stride):
    contexts, targets, anchors = [], [], []
    for anchor in range(history-1, len(motion.joints)-horizon, stride):
        before = slice(anchor-history+1,anchor+1)
        after = slice(anchor+1,anchor+horizon+1)
        origin = motion.root_p[anchor]
        inv = motion.root_r[anchor].inv()

        def pose(indices):
            return np.column_stack((motion.joints[indices], inv.apply(motion.root_p[indices]-origin),
                                    (inv*motion.root_r[indices]).as_rotvec()))

        past = pose(before)
        future = pose(after)
        future[:,:29] -= motion.joints[anchor]
        contexts.append(np.r_[past.ravel(), float(motion.side=='left'),float(motion.side=='right')])
        targets.append(future)
        anchors.append(anchor)
    return Windows(np.asarray(contexts),np.asarray(targets),np.asarray(anchors))


@dataclass
class Norm:
    xmu: np.ndarray
    xsd: np.ndarray
    ymu: np.ndarray
    ysd: np.ndarray

    @classmethod
    def fit(cls, training):
        x = np.concatenate([w.context for w in training])
        y = np.concatenate([w.target for w in training]).reshape(-1,35)
        return cls(x.mean(0), np.maximum(.05,x.std(0)), y.mean(0), np.maximum(.025,y.std(0)))

    def x(self, values):
        return torch.as_tensor((values-self.xmu)/self.xsd,dtype=torch.float32)

    def y(self, values):
        values=(values-self.ymu)/self.ysd
        return torch.as_tensor(values.reshape(len(values),-1),dtype=torch.float32)

    def decode(self, tensor):
        return tensor.detach().numpy().reshape(len(tensor),-1,35)*self.ysd+self.ymu


def mlp(size_in,width,size_out):
    return nn.Sequential(nn.Linear(size_in,width),nn.ELU(),nn.Linear(width,size_out))


class ConditionalVAE(nn.Module):
    def __init__(self,xsize,ysize,hidden,latent):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(xsize,hidden),nn.ELU())
        self.prior=nn.Linear(hidden,2*latent)
        self.posterior=mlp(hidden+ysize,hidden,2*latent)
        self.decoder=mlp(hidden+latent,hidden,ysize)

    @staticmethod
    def normal(params):
        mean,logvar=params.chunk(2,-1)
        return mean,logvar.clamp(-7.,4.)

    def zprior(self,x):
        h=self.context(x)
        return h,*self.normal(self.prior(h))

    def decode(self,h,z):
        return self.decoder(torch.cat((h,z),dim=-1))

    def training_loss(self,x,y,beta,noise_generator):
        h,prior_mean,prior_logvar=self.zprior(x)
        posterior_mean,posterior_logvar=self.normal(self.posterior(torch.cat((h,y),dim=-1)))
        epsilon=torch.randn(posterior_mean.shape,generator=noise_generator)
        z=posterior_mean+torch.exp(.5*posterior_logvar)*epsilon
        pred=self.decode(h,z)
        kl=.5*(prior_logvar-posterior_logvar-1.+torch.exp(posterior_logvar-prior_logvar)
               +(posterior_mean-prior_mean).square()*torch.exp(-prior_logvar)).mean()
        return (pred-y).square().mean()+beta*kl


def count(model):
    return sum(p.numel() for p in model.parameters())


def matching_width(xsize,ysize,nparams):
    return min(range(1,1025), key=lambda w: abs((xsize+1)*w+(w+1)*ysize-nparams))


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def position_velocity_limits(path):
    joints={j.get('name'):j.find('limit') for j in ET.parse(path).getroot().findall('joint')}
    return np.array([[float(joints[name].get(key)) for key in ('lower','upper','velocity')] for name in G1_JOINT_ORDER])


def body_groups(side):
    groups={}
    for limb,prefix in [('left_arm','left'),('right_arm','right'),('left_leg','left'),('right_leg','right')]:
        parts=('shoulder','elbow','wrist') if limb.endswith('arm') else ('hip','knee','ankle')
        groups[limb]=[i for i,n in enumerate(G1_JOINT_ORDER) if n.startswith(prefix+'_') and any('_'+p+'_' in n for p in parts)]
    groups['waist']=[i for i,n in enumerate(G1_JOINT_ORDER) if n.startswith('waist_')]
    if side in ('left','right'):
        groups['labeled_kicking_leg']=groups[side+'_leg']
        groups['opposite_leg']=groups[('right' if side=='left' else 'left')+'_leg']
    return groups


def reference_violations(delta, initial_q, limits):
    q=delta[:,:,:29]+initial_q[:,None,:]
    qd=np.diff(q,axis=1,prepend=initial_q[:,None,:])*50.
    pose=(q<limits[:,0]-1e-6)|(q>limits[:,1]+1e-6)
    speed=np.abs(qd)>limits[:,2]+1e-6
    return pose,speed


def angle_errors(a,b):
    return (Rotation.from_rotvec(a.reshape(-1,3)).inv()*Rotation.from_rotvec(b.reshape(-1,3))).magnitude()


def scores(pred,w,motion,limits):
    target=w.target
    diff=np.abs(pred[:,:,:29]-target[:,:,:29])
    tr, sr=reference_violations(target,motion.joints[w.anchors],limits)
    qp, vp=reference_violations(pred,motion.joints[w.anchors],limits)
    return dict(joint_mae_rad=float(diff.mean()),root_displacement_mae_m=float(np.linalg.norm(pred[:,:,29:32]-target[:,:,29:32],axis=-1).mean()),
                root_angle_mae_rad=float(angle_errors(pred[:,:,32:],target[:,:,32:]).mean()),
                group_mae_rad={name:float(diff[:,:,ids].mean()) for name,ids in body_groups(motion.side).items()},
                source_pose_violations=int(tr.sum()),source_speed_violations=int(sr.sum()),
                generated_pose_violations=int(qp.sum()),generated_speed_violations=int(vp.sum()),
                new_pose_violations=int((qp&~tr).sum()),new_speed_violations=int((vp&~sr).sum()),
                evaluated_joint_values=int(qp.size))


def max_true_run(flags):
    best=run=0
    for x in flags:
        run=run+1 if x else 0
        best=max(run,best)
    return best


def intervention_stats(plus,minus,motion):
    diff=np.abs(plus[:,:,:29]-minus[:,:,:29]); out={}
    for name,ids in body_groups(motion.side).items():
        by_window=np.sqrt((diff[:,:,ids]**2).mean(-1))
        out[name]=dict(rms_rad=float(np.sqrt(np.mean(diff[:,:,ids]**2))),
                       fraction_windows_above_1mrad_for_3_frames=float(np.mean([max_true_run(x>.001)>=3 for x in by_window])))
    out['pelvis']=dict(displacement_difference_m=float(np.linalg.norm(plus[:,:,29:32]-minus[:,:,29:32],axis=-1).mean()),
                        orientation_difference_rad=float(angle_errors(plus[:,:,32:],minus[:,:,32:]).mean()))
    return out


def hold_and_extrapolation(w,history,horizon):
    past=w.context[:,:-2].reshape(-1,history,35)
    last_step=np.concatenate((past[:,-1,:29]-past[:,-2,:29], -past[:,-2,29:32],-past[:,-2,32:35]),axis=-1)
    extrap=last_step[:,None,:]*np.arange(1,horizon+1)[None,:,None]
    return np.zeros_like(extrap),extrap


def reversed_context(w,history):
    pre=w.context[:,:-2].reshape(-1,history,35)
    reverted=np.concatenate((pre[:,:-1][:,::-1],pre[:,-1:]),axis=1)
    return np.column_stack((reverted.reshape(len(pre),-1),w.context[:,-2:]))


@torch.no_grad()
def evaluate(cv,det,norm,w,motion,limits,cfg,seed):
    x=norm.x(w.context);h,m,lv=cv.zprior(x)
    generator=torch.Generator().manual_seed(seed+7907)
    eps=torch.randn(m.shape,generator=generator)
    stride=max(1,len(x)//2)
    hrev,mrev,_=cv.zprior(norm.x(reversed_context(w,cfg['past_frames'])))
    values={'cvae_prior_mean':norm.decode(cv.decode(h,m)), 'deterministic':norm.decode(det(x)),
            'zero_z':norm.decode(cv.decode(h,torch.zeros_like(m))),
            'shifted_z_within_clip':norm.decode(cv.decode(h,torch.roll(m,stride,0))),
            'reversed_history':norm.decode(cv.decode(hrev,mrev)),
            'prior_sample_plus':norm.decode(cv.decode(h,m+eps*torch.exp(.5*lv))),
            'prior_sample_minus':norm.decode(cv.decode(h,m-eps*torch.exp(.5*lv)))}
    values['hold'],values['velocity_extrapolation']=hold_and_extrapolation(w,cfg['past_frames'],cfg['future_frames'])
    metrics={k:scores(v,w,motion,limits) for k,v in values.items()}
    return dict(windows=len(x),metrics=metrics,latent_intervention=intervention_stats(values['prior_sample_plus'],values['prior_sample_minus'],motion)),values


def ensure_lock(commit, config_path):
    if len(commit)!=40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('Pass the full pre-training Git commit SHA')
    for path in [Path(__file__),config_path,Path(__file__).with_name('wholebody_safety_interface_v055.py')]:
        rel=str(path.resolve().relative_to(ROOT))
        content=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=sha(path):raise ValueError('Source differs from pre-training commit: '+rel)


def run(source,cfg,assets,lock,output):
    setting=cfg['sources'][source]; splits=clip_splits(setting)
    schema_path=assets/cfg['assets']['runtime_joint_and_body_schema'];urdf=assets/cfg['assets']['urdf']
    if sha(schema_path)!=cfg['assets']['runtime_joint_and_body_schema_sha256'] or sha(urdf)!=cfg['assets']['urdf_sha256']:
        raise ValueError('Original 29DoF schema or URDF changed')
    schema=json.loads(schema_path.read_text());limits=position_velocity_limits(urdf)
    if list(G1_JOINT_ORDER)!=schema['joint_names'] or schema['body_names'][0]!='pelvis':raise ValueError('Runtime shape/order mismatch')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    if source=='cmu':
        folder=ROOT/setting['relative_root'];mpath=folder/setting['manifest'];manifest=json.loads(mpath.read_text())
        source_hash=sha(mpath)
    else:
        source_hash=None
    motions={}; windows={};clip_info={}
    for part,names in splits.items():
        windows[part]={}
        for name in names:
            if source=='paid':
                path=assets/setting['paths'][name];expected=setting['hashes'][name]
            else:
                path=folder/(name+'.npz');entry=manifest['clips'][name];expected=entry['npz_sha256']
                if str(entry['subject_id'])!=setting['source_subjects'][name]:raise ValueError('CMU subject metadata mismatch')
                if not entry.get('source_c3d_sha256') or not entry.get('source_processed_sha256'):
                    raise ValueError('Incomplete original human source contract')
            motion=load_motion(path,name,source,schema,expected);motions[name]=motion
            stride=cfg['train_stride'] if part=='train' else cfg['evaluation_stride']
            windows[part][name]=make_windows(motion,cfg['past_frames'],cfg['future_frames'],stride)
            clip_info[name]=dict(split=part,frames=len(motion.joints),windows=len(windows[part][name].anchors),
                                 motion_sha256=motion.sha256,kick_side=motion.side,player_id=setting.get('source_subjects',{}).get(name))
    train=list(windows['train'].values()); norm=Norm.fit(train)
    train_x=[norm.x(w.context) for w in train];train_y=[norm.y(w.target) for w in train]
    xsize=train_x[0].shape[1];ysize=train_y[0].shape[1]
    out=dict(offline_reference_only=True,source=source,source_manifest_sha256=source_hash,pretraining_commit=lock,
             source_files=clip_info,torch_version=str(torch.__version__),python_version=platform.python_version(),
             conditioning=cfg['rollout_conditions'],normalization_fit_split='train',
             independent_test_clips=len(splits['test']),independent_test_players=1 if source=='cmu' else None,
             contact_ground_truth_available=False,physical_rollouts=0,records=[])
    np.savez_compressed(output/'training_normalization.npz',**vars(norm))
    timing=time.monotonic()
    for seed in cfg['training_seeds']:
        torch.manual_seed(seed);cv=ConditionalVAE(xsize,ysize,cfg['hidden_width'],cfg['latent_dim'])
        width=matching_width(xsize,ysize,count(cv))
        torch.manual_seed(seed);det=mlp(xsize,width,ysize)
        noise=torch.Generator().manual_seed(seed+19);rng=np.random.default_rng(seed+37)
        optimizers=[torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']) for net in (cv,det)]
        for step in range(cfg['steps']):
            selected=rng.integers(len(train),size=cfg['batch_size'])
            pairs=[(i,rng.integers(len(train_x[i]))) for i in selected]
            x=torch.stack([train_x[i][j] for i,j in pairs]);y=torch.stack([train_y[i][j] for i,j in pairs])
            losses=(cv.training_loss(x,y,cfg['kl_weight'],noise),(det(x)-y).square().mean())
            for optimizer,loss in zip(optimizers,losses):
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite optimization objective')
                optimizer.zero_grad();loss.backward();optimizer.step()
        cv.eval();det.eval()
        if any(not bool(t.isfinite().all()) for net in (cv,det) for t in net.state_dict().values()):raise FloatingPointError('Nonfinite final state')
        ckpt=output/f'models_seed{seed}.pt'
        torch.save({'cvae':cv.state_dict(),'deterministic':det.state_dict(),'deterministic_hidden':width,'seed':seed},ckpt)
        record=dict(seed=seed,checkpoint_sha256=sha(ckpt),parameters={'cvae_total':count(cv),'cvae_deployed':count(cv)-count(cv.posterior),'deterministic_total':count(det),'deterministic_width':width},splits={})
        for part in ('validation','test'):
            per_clip={}
            for name,w in windows[part].items():
                metrics,values=evaluate(cv,det,norm,w,motions[name],limits,cfg,seed)
                per_clip[name]=metrics
                if part=='test':
                    np.savez_compressed(output/f'forecast_{name}_seed{seed}.npz',anchor_frame=w.anchors,
                                        target_delta=w.target,joint_names=G1_JOINT_ORDER,**values)
            model_names=next(iter(per_clip.values()))['metrics'].keys()
            means={model:float(np.mean([x['metrics'][model]['joint_mae_rad'] for x in per_clip.values()])) for model in model_names}
            record['splits'][part]={'clips':per_clip,'equal_clip_joint_mae_rad':means}
        out['records'].append(record)
        print('finished',source,seed,record['splits']['test']['equal_clip_joint_mae_rad'],flush=True)
    rows=[r['splits']['test']['equal_clip_joint_mae_rad'] for r in out['records']]
    out['test_mean_over_seeds']={name:float(np.mean([r[name] for r in rows])) for name in rows[0]}
    out['beats_mlp_and_both_simple_forecasters_every_seed']=all(r['cvae_prior_mean']<min(r[k] for k in ('deterministic','hold','velocity_extrapolation')) for r in rows)
    out['training_and_evaluation_wall_seconds']=time.monotonic()-timing
    write_json(output/'results.json',out)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['paid','cmu'],required=True)
    parser.add_argument('--asset-root',type=Path,default=Path('/home/you/projects/kinetickick'))
    parser.add_argument('--frozen-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ensure_lock(args.frozen_commit,CONFIG)
    cfg=json.loads(CONFIG.read_text());torch.set_num_threads(cfg['num_cpu_threads']);torch.use_deterministic_algorithms(True)
    run(args.source,cfg,args.asset_root,args.frozen_commit,args.output)


if __name__=='__main__':main()
