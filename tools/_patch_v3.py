from pathlib import Path
p=Path("/home/you/projects/kinetickick/tools/eval_free_root_walk_to_kick_v3.py")
s=p.read_text()
changes=[
('''BALL_X=(1.22,1.30,1.38,1.46)
AMPLITUDES=(.30,.50,.75)
CASES=[{"ball_x":bx,"ball_y":-.11,"magnitude":amp,
        "name":f"bx{bx:.2f}_a{amp:.2f}"}
       for bx in BALL_X for amp in AMPLITUDES]''',
'''SUPPORT_BRACE=(0.0,.18,.36,.52)
AMPLITUDES=(.28,.44,.62)
CASES=[{"ball_x":1.22,"ball_y":-.11,"magnitude":amp,"brace":brace,
        "name":f"bx1.22_amp{amp:.2f}_brace{brace:.2f}"}
       for brace in SUPPORT_BRACE for amp in AMPLITUDES]'''),
('''    "right_hip_roll_joint","left_hip_roll_joint",''',
'''    "right_hip_roll_joint","left_hip_roll_joint",
    "left_knee_joint","left_hip_pitch_joint","left_ankle_pitch_joint",'''),
('''PHASE_DURATIONS={"preload":.46,"strike":.32,"follow_through":.34,"recover":.80}''',
'''PHASE_DURATIONS={"preload":.46,"strike":.32,"follow_through":.10,"recover":.26}'''),
('''def desired_residual(phase,elapsed,strength,hip_shift=0.08):''',
'''def desired_residual(phase,elapsed,strength,brace=0.,hip_shift=0.08):'''),
('''        "left_hip_roll_joint":m*hip_shift*lean,''',
'''        "left_hip_roll_joint":m*hip_shift*lean,
        "left_knee_joint":-brace*lean,
        "left_hip_pitch_joint":.12*brace*lean,
        "left_ankle_pitch_joint":-.06*brace*lean,'''),
('''delta=desired_residual(ph,s["elapsed"],case["magnitude"])''',
'''delta=desired_residual(ph,s["elapsed"],case["magnitude"],case["brace"])'''),
('''                if s["stage"] in ("strike","follow_through"):
                    s["first_kick_contact"]=dict(s["first_contact"])''',
'''                if s["stage"] in ("strike","follow_through"):
                    s["first_kick_contact"]=dict(s["first_contact"])
                    if s["stage"]=="strike":
                        s["stage"]="follow_through";s["elapsed"]=0.
                        print("FREE_ROOT_CONTACT_TRIGGERED_EARLY_RECOVERY",i,case["name"],step,flush=True)'''),
('''    ap.add_argument("--output",type=str,default="reports/shot_free_root_multitrial_v1.json")''',
'''    ap.add_argument("--output",type=str,default="reports/shot_free_root_multitrial_v3.json")''')
]
for old,new in changes:
    assert old in s, old
    s=s.replace(old,new,1)
p.write_text(s)
print("V3_PATCHED",p,"length",len(s))
