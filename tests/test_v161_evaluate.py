import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from v161_evaluate import patch_summary,measured_follow,reconstruct_com,episode_readout,tail_duration


def test_two_first_contact_points_count_ambiguous_impulse_in_denominator():
    times=np.arange(1,5)*.005;foot=np.zeros((4,3));rot=Rotation.identity(4)
    ball=np.tile([.06,0.,.09],(4,1))
    top=np.array([.06,0.,-.02, 0.,0.,1., 0.,0.,.05,0.])
    foreign=np.array([.9,0.,-.02, 0.,0.,1., 0.,0.,.05,0.])
    raw=[(2,top,1,.05,'right/shape[1]','ball/shape[0]'),(2,foreign,1,.05,'right/shape[2]','ball/shape[0]')]
    score=patch_summary(raw,times,foot,rot,ball,.11,2,2)
    assert score['instep_fraction']==.5
    assert score['per_patch_normal_impulse_Ns']['ambiguous']==.05


def test_missing_shot_and_final_standing_do_not_require_an_event():
    t=np.arange(1,201)*.005;body=np.zeros((200,3))
    assert not measured_follow(t,body,body,None,np.array([0.,-1.,0.]),np.zeros(200,bool))['passed']
    assert episode_readout(np.zeros(200,bool),t,body[:,:2])==[]
    assert tail_duration(np.array([True,False,True,True,True]),.005)==.01


def test_terminal_exit_follows_last_loaded_interval():
    t=np.arange(1,7)*.005
    vel=np.array([[0.,x] for x in [0,1,2.2,2.7,4.9,4.4]])
    endpoints=episode_readout(np.array([False,True,False,True,True,False]),t,vel)
    assert len(endpoints)==2 and endpoints[0]['velocity_xy_mps_after_interval']==2.2
    assert endpoints[-1]['velocity_xy_mps_after_interval']==4.4


def test_com_includes_unrecorded_fixed_masses_and_rotated_inertias():
    xml=ET.fromstring('<robot><link name="root"><inertial><mass value="1"/></inertial></link><link name="fixed"><inertial><origin xyz="1 0 0"/><mass value="2"/></inertial></link><joint name="f" type="fixed"><parent link="root"/><child link="fixed"/><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/></joint></robot>')
    q=Rotation.from_euler('z',[0.,np.pi/2]).as_quat()[:,[3,0,1,2]][:,None,:]
    com,mass=reconstruct_com(xml,['root'],np.zeros((2,1,3)),q)
    assert mass==3 and np.allclose(com,[[2/3,2/3,0],[-2/3,2/3,0]],atol=1e-12)
