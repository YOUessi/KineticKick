"""Restore actor and exploration state while keeping a fresh critic/optimizer."""
import torch


def restore_actor(algorithm,path,device):
    saved=torch.load(path,map_location=device,weights_only=False)['model_state_dict']
    target=algorithm.policy.state_dict()
    for key in target:
        if key.startswith('actor.') or key=='std':
            if key not in saved or saved[key].shape!=target[key].shape:raise ValueError('Warmstart actor shape differs: '+key)
            target[key]=saved[key]
    algorithm.policy.load_state_dict(target)
