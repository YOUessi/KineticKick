#!/usr/bin/env python3
"""Isolated yawed sliding-body impulse check for normal/friction force telemetry."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject,RigidObjectCfg
        from pxr import UsdPhysics
        import omni.usd
        from strike_dynamics_telemetry import contact_components,certify_force_probe
        simulation=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,render_interval=4,device='cuda:0'))
        material=sim_utils.RigidBodyMaterialCfg(static_friction=.6,dynamic_friction=.5,restitution=0.)
        ground_cfg=sim_utils.GroundPlaneCfg(physics_material=material);ground_cfg.func('/World/Ground',ground_cfg)
        body=RigidObject(RigidObjectCfg(prim_path='/World/Probe',spawn=sim_utils.CuboidCfg(size=(.2,.2,.2),activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),collision_props=sim_utils.CollisionPropertiesCfg(),mass_props=sim_utils.MassPropertiesCfg(mass=1.),physics_material=material),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.102),rot=(math.cos(math.pi/8),0.,0.,math.sin(math.pi/8)),lin_vel=(1.2,0.,0.))))
        simulation.reset();body.write_root_state_to_sim(body.data.default_root_state.clone());body.reset();body.update(0.)
        ground=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse() if str(p.GetPath()).startswith('/World/Ground/') and p.HasAPI(UsdPhysics.CollisionAPI)]
        if not ground:raise ValueError('No ground collision primitive')
        view=body._physics_sim_view.create_rigid_contact_view('/World/Probe',ground,max_contact_data_count=64)
        initial=body.data.root_lin_vel_w[0].cpu().tolist();mass=float(body.root_physx_view.get_masses().sum());rows=[]
        for i in range(240):
            body.write_data_to_sim();simulation.step(render=False);body.update(.005)
            sample=contact_components(view,.005);sample.update(time_s=(i+1)*.005,dt_s=.005,velocity_w=body.data.root_lin_vel_w[0].cpu().tolist());rows.append(sample)
        certificate=certify_force_probe(rows,mass,initial,[0.,0.,-9.81])
        result=dict(scope='Isolated1kg yawed sliding cube; no robot/ball or research-controller modification',mass_kg=mass,initial_velocity_w=initial,gravity_w=[0.,0.,-9.81],ground=ground,samples=rows,certificate=certificate,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tools/strike_dynamics_telemetry.py']})
        args.output.write_text(json.dumps(result,indent=2)+'\n');print('FORCE_PROBE',json.dumps(certificate),flush=True)
        if not certificate['verified']:raise RuntimeError('Force semantics probe did not pass')
    except BaseException:
        import traceback
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
    finally:
        # Match ManagerBasedEnv's order-sensitive shutdown: release tensor
        # observers and SimulationContext callbacks before closing Kit.
        if 'view' in locals():del view
        if 'body' in locals():del body
        if 'simulation' in locals():
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        app.close()

if __name__=='__main__':main()
