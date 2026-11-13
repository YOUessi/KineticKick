"""Local inverse dynamics calibration: identify a command map near a single physical swing."""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
from v162_grid import feasible


def query(worker,x):
    worker.stdin.write(json.dumps(x.tolist())+'\n');worker.stdin.flush()
    return json.loads(worker.stdout.readline())


def observed(row):
    if row['release_xy'] is None or row['crossing_x'] is None:return None
    return np.array([np.linalg.norm(row['release_xy']), row['crossing_x']])


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--seed',type=Path,default=Path('configs/v161_margin_s0.json'));cli.add_argument('--rng',type=int,default=162440)
    cli.add_argument('--radius',type=float,default=.025)
    args=cli.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.seed.read_text());x=np.array(config['coefficients']);x=np.r_[x[:12],config.get('swing_lead_s',0.),config.get('swing_rate',1.),config.get('hip_drive',0.)]
    cap=config['backswing_pitch_target_cap']
    worker=subprocess.Popen([sys.executable,str(Path(__file__).with_name('v161_dynamics_search.py')),'--live','--feasible','--service','--pitch-cap',str(cap),'--output',str(args.output/'worker')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        if worker.stdout.readline().strip()!='ready':raise RuntimeError('Physics startup failed')
        steps=np.tile([1,.7,.8,1,.7,.5],2).tolist()+[.045,.5,1.]
        scales=np.array(steps)*args.radius
        rng=np.random.default_rng(args.rng)
        D=np.vstack([np.zeros(15),np.eye(15),-np.eye(15),rng.normal(size=(370,15))])
        total_trials=len(D)
        samples=[]
        with (args.output/'calibration.jsonl').open('x') as f:
            for delta in D:
                params=x+delta*scales;row=query(worker,params);y=observed(row)
                f.write(json.dumps({'coefficients':params.tolist(),**row})+'\n')
                if y is not None and feasible(row):samples.append((delta,y))
        D,Y=map(np.array,zip(*samples));origin=np.c_[D,np.ones(len(D))]
        beta=np.linalg.lstsq(origin,Y,rcond=.0001)[0];jac=beta[:-1].T
        inv=jac.T@np.linalg.inv(jac@jac.T+np.diag([.0001,.00005]))
        res=Y-origin@beta
        domain={'speed':[3.55,4.30],'aim':[-.26,.30]}
        obj={'module':'v162_strike_power','map_type':'local_inverse','base':x.tolist(),'observation_mean':beta[-1].tolist(),
             'scaled_inverse':(scales[:,None]*inv).tolist(),'absolute_delta_limit':(3.5*scales).tolist(),'domain':domain,
             'backswing_pitch_target_cap':cap,'wrist_clearance':.22,
             'calibration_rows':len(D),'total_rows':total_trials,'rms':np.sqrt(np.mean(res**2,axis=0)).tolist(),
             'calibration_sha256':hashlib.sha256((args.output/'calibration.jsonl').read_bytes()).hexdigest()}
        (args.output/'map.json').write_text(json.dumps(obj,indent=2)+'\n');print(len(D),obj['observation_mean'],obj['rms'],np.linalg.svd(jac)[1])
    finally:
        worker.stdin.close();worker.wait()
if __name__=='__main__':main()
