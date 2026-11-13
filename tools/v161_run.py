"""Run a measured v160 CPU scene with independently switchable control interventions."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import v160_tracking_cpu as base


def load_controller(path, kind):
    if path is None:
        return None
    config = json.loads(path.read_text())
    name = config['module']
    if not name.isidentifier() or not (name.startswith(('v161_' + kind, 'v162_' + kind)) or (kind == 'strike' and name in {'v162_vertical','v162_moving'})):
        raise ValueError(f'Unexpected {kind} controller module {name}')
    module = importlib.import_module(name)
    return module, config, {'file':str(path), 'config':config, 'sha256':base.digest(path),
                            'code_sha256':base.digest(Path(module.__file__))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--assets', type=Path, default=base.DEFAULT_ASSETS)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--strike-config', type=Path)
    cli.add_argument('--recovery-config', type=Path)
    refs=cli.add_mutually_exclusive_group()
    refs.add_argument('--clip')
    refs.add_argument('--reference-file',type=Path,help='Complete human-derived 50 Hz G1 trajectory with named axes')
    cli.add_argument('--duration', type=float, default=14.)
    cli.add_argument('--ball-offset', nargs=2, type=float, default=[0.,0.], metavar=('X','Y'))
    cli.add_argument('--ball-velocity', nargs=2, type=float, default=[0.,0.], metavar=('VX','VY'),
                     help='Initial world-frame rolling velocity in m/s')
    args = cli.parse_args()
    cfg = json.loads(base.PROTOCOL.read_text())
    if not 0 < args.duration <= 14 or not np.isclose(args.duration*50,round(args.duration*50)):
        raise ValueError('Duration must consist of whole policy steps up to 14 seconds')
    cfg['duration_s'] = args.duration
    if args.clip is not None:
        if args.clip not in cfg['motion_sha256']:
            raise ValueError('Expected a qualified reference clip from the frozen PAiD inventory')
        cfg['clip'] = args.clip
    cfg['ball_initial_xyz_m'][:2] = (np.array(cfg['ball_initial_xyz_m'][:2])+args.ball_offset).tolist()
    if not np.isfinite([*args.ball_offset, *args.ball_velocity]).all():
        cli.error('Initial ball offset and velocity must be finite')
    cfg['ball_initial_velocity_xy_mps'] = args.ball_velocity
    assets = args.assets.resolve()
    for path, wanted in {**cfg['asset_hashes'],cfg['clip']:cfg['motion_sha256'][cfg['clip']]}.items():
        if base.digest(assets/path)!=wanted:
            raise ValueError('Changed asset '+path)
    schema=json.loads((assets/base.SCHEMA).read_text())
    external={}
    if args.reference_file is not None:
        file=args.reference_file.resolve(strict=True)
        with np.load(file,allow_pickle=False) as data:
            frames=len(data['joint_pos'])
            shapes={'joint_pos':(frames,29),'joint_vel':(frames,29),
                    'body_pos_w':(frames,30,3),'body_quat_w':(frames,30,4),
                    'body_lin_vel_w':(frames,30,3),'body_ang_vel_w':(frames,30,3)}
            for key,shape in shapes.items():
                if data[key].shape!=shape or not np.isfinite(data[key]).all():
                    raise ValueError('Invalid human reference '+key)
            if (frames<2 or data['fps'].tolist()!=[50] or
                data['joint_names'].tolist()!=schema['joint_names'] or data['body_names'].tolist()!=schema['body_names'] or
                not np.allclose(data['time_s'],np.arange(frames)/50,rtol=0,atol=1e-7) or
                not np.allclose(np.linalg.norm(data['body_quat_w'],axis=-1),1,atol=1e-4)):
                raise ValueError('Invalid human reference names, time or rotations')
        cfg['clip']=str(file)
        external={'external_reference_sha256':base.digest(file)}
    controls=[x for x in (load_controller(args.strike_config,'strike'),
                           load_controller(args.recovery_config,'recovery')) if x]
    args.output.mkdir(parents=True,exist_ok=False)
    base.save_json(args.output/'trial.json',dict(
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),
        runtime_code_sha256=base.digest(Path(base.__file__)),runner_sha256=base.digest(Path(__file__)),
        controllers=[c[2] for c in controls], duration_s=args.duration, **external,
        ball_initial_velocity_xy_mps=cfg['ball_initial_velocity_xy_mps'],
        ball_initial_xyz_m=cfg['ball_initial_xyz_m'],clip=cfg['clip']))
    def factory(ref,schema):
        funcs=[]
        for module,config,_ in controls:
            if hasattr(module,'modify_reference'):
                edited=module.modify_reference(ref,schema,config)
                if edited is not None:
                    ref=edited
            funcs.append(module.make_controller(ref,schema,config))
        def control(ctx,target):
            for fn in funcs:
                target=fn(ctx,target)
            return target
        return ref,control
    with base.cpu_policy(assets,cfg) as (meta,act):
        if meta['body_names'].split(',')!=cfg['tracking_links']:
            raise ValueError('Unexpected PAiD body order')
        base.run(assets,args.output,cfg,schema,meta,act,controller_factory=factory)


if __name__=='__main__':
    main()
