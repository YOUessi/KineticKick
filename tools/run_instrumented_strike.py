#!/usr/bin/env python3
"""Observe original v15/v28 physical runners without changing their controls."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controller',choices=['v15','v28','placement'],default='v28')
    p.add_argument('--config',type=Path,help='Required only for placement research controllers')
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ground-contact-window',type=float,nargs=2,help='Optional bounded raw ground contact/anchor and pre/post foot evidence')
    p.add_argument('--probe',type=Path,default=ROOT/'reports/v029_force_probe.json');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if (a.controller=='placement') != (a.config is not None):p.error('--config is required only with --controller placement')
    config=a.config if a.config is not None else ROOT/'configs/state_conditioned_strike_v5.json'
    from strike_cases import load_cases
    cases=load_cases(ROOT)
    if a.case not in cases:p.error('Unknown registered condition')
    if not json.loads(a.probe.read_text())['certificate']['verified']:raise ValueError('Force probe required')
    original_spec=importlib.util.spec_from_file_location;source=ROOT/'tools/eval_free_shooting.py'
    def instrumented_spec(name,location,*aa,**kk):
        spec=original_spec(name,location,*aa,**kk)
        if Path(location).resolve()!=source:return spec
        native_load=spec.loader.exec_module
        def load(module):
            native_load(module);native_run=module.run
            def observed_run(runtime_args,points):
                import gymnasium as gym
                from strike_dynamics_telemetry import StrikeDynamicsObserver
                native_make=gym.make;created={}
                def make(*pos,**kw):
                    env=native_make(*pos,**kw);raw=env.unwrapped
                    created.update(raw=raw,scene=raw.scene)
                    try:
                        if a.ground_contact_window is None:observer=StrikeDynamicsObserver(raw,a.probe)
                        else:
                            from ground_contact_observer import GroundContactObserver
                            observer=GroundContactObserver(raw,a.probe,a.ground_contact_window)
                        created['observer']=observer
                        native_update=raw.scene.update;native_step=raw.step
                        created.update(native_update=native_update,native_step=native_step)
                        def update(dt):
                            native_update(dt);observer.after_update(dt)
                        def step(actions):
                            observer.active=True
                            return native_step(actions)
                        raw.scene.update=update;raw.step=step
                    except BaseException:
                        if 'native_update' in created:created['scene'].update=created['native_update']
                        if 'native_step' in created:raw.step=created['native_step']
                        env.close()
                        raise
                    return env
                gym.make=make
                try:
                    value=native_run(runtime_args,points)
                    result=json.loads(a.output.read_text());created['observer'].attach(result)
                    result['dynamics_telemetry'].update(runner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controller=a.controller,registered_case=a.case)
                    a.output.write_text(json.dumps(result,indent=2)+'\n')
                    print('DYNAMICS_TELEMETRY',json.dumps(dict(samples=len(created['observer'].rows),metadata=result['dynamics_telemetry'])),flush=True)
                    return value
                finally:
                    gym.make=native_make
                    if 'native_update' in created:created['scene'].update=created['native_update']
                    if 'native_step' in created:created['raw'].step=created['native_step']
            module.run=observed_run
        spec.loader.exec_module=load
        return spec
    path=ROOT/'tools'/('run_placement_strike.py' if a.controller!='v15' else 'run_strike_baseline_case.py')
    spec=original_spec('_observed_original_runner',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    argv=sys.argv
    try:
        importlib.util.spec_from_file_location=instrumented_spec
        if a.controller!='v15':
            c=cases[a.case]
            sys.argv=[str(path),'--config',str(config),'--output',str(a.output),'--ball-offset',*map(str,c['ball_offset_xy_m']),'--velocity-delta',str(c['root_velocity_y_delta_mps']),'--contact-diagnostics']
        else:sys.argv=[str(path),'--case',a.case,'--output',str(a.output)]
        runner.main()
    finally:importlib.util.spec_from_file_location=original_spec;sys.argv=argv

if __name__=='__main__':main()
