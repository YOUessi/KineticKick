"""Freeze the best physically qualified development trace per anchor; keep diverse solutions explicit."""
from pathlib import Path
import argparse,json,hashlib

def feasible(r):
    return (min(r['instep'],r['instep_post'])>=.95 and r['excess']<=1e-6 and
            r['nonfoot']<20 and not r['late_contact'] and r['return_speed']<=2 and
            r['advance']>=.22 and r['relative']>=.15 and r['v120']>=.6 and
            r['crossing_x'] is not None and r['release_xy'] is not None)

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--sources',nargs='+',type=Path,required=True);cli.add_argument('--output',required=True,type=Path);a=cli.parse_args()
    proto=json.loads(Path('configs/v162_power.json').read_text());speeds,aims=proto['development_speed_mps'],proto['development_aim_x_m']
    grid=[];chosen=[]
    for i,speed in enumerate(speeds):
        row=[]
        for j,aim in enumerate(aims):
            options=[]
            for source in a.sources:
                f=source/f'train{i}{j}'/'development.jsonl'
                if not f.is_file():continue
                options.extend((r,str(f)) for r in map(json.loads,f.read_text().splitlines()) if feasible(r))
            if not options:raise ValueError(f'No eligible development candidate for {speed}, {aim}')
            entry,file=min(options,key=lambda item:item[0]['loss'])
            coeff=entry['coefficients']
            if len(coeff)==12:coeff=coeff+[0.,1.,0.]
            row.append(coeff)
            chosen.append({'command':[speed,aim],'loss':entry['loss'],'round':entry['round'],'ledger':file,
                           'sha256':hashlib.sha256(Path(file).read_bytes()).hexdigest()})
        grid.append(row)
    config=dict(module='v162_strike_power',speeds_mps=speeds,aims_m=aims,coefficient_grid=grid,
                backswing_pitch_target_cap=.9,wrist_clearance=.22,selected=chosen)
    with a.output.open('x') as o:json.dump(config,o,indent=2);o.write('\n')
if __name__=='__main__':main()
