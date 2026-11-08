#!/usr/bin/env python3
"""v39: audited professional-football *broadcast-estimated* 3D coordination.

Archive 3DSP (CVPR 2024 AutoSoccerPose), 200 training shots x20 predicted
3D keypoints, not calibrated high-speed MoCap. All 3D "skeleton" coordinates
are estimated from 2D broadcast and metric scale/impact time are unverified.
Features are similarity-invariant SEGMENT DIRECTION COSINES, not actual
anatomical joint angles, ground forces, power or human-like robot actions.

Never unzips images, never imports unsafe pickle; source data remain ignored.
"""
from __future__ import annotations
from pathlib import Path
import argparse,configparser,hashlib,io,json,math,random,re,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE_REL="data/research_human_kicks/3dsp_cvpr2024/3dsp.zip"
SOURCE_SHA="3696523b69dd3033435f76ab64410aa8e34172df32bb9673eca93b6609c87ee7"
SOURCE_GIT="c34f46d2f5d42e996ee328a95ad49217206b5607"

JOINT_ORDER={
 "Center of Hips":0,"Left Hip":1,"Left Knee":2,"Left Ankle":3,
 "Right Hip":4,"Right Knee":5,"Right Ankle":6,
 "Center of Body":7,"Center of Shoulder":8,
 "Neck":9,"Head":10,"Right Shoulder":11,"Right Elbow":12,
 "Right Wrist":13,"Left Shoulder":14,"Left Elbow":15,"Left Wrist":16
}
SEGMENTS={
 "trunk":("Center of Hips","Center of Shoulder"),
 "pelvis_line":("Right Hip","Left Hip"),
 "shoulder_line":("Right Shoulder","Left Shoulder"),
 "left_thigh":("Left Hip","Left Knee"),
 "right_thigh":("Right Hip","Right Knee"),
 "left_shank":("Left Knee","Left Ankle"),
 "right_shank":("Right Knee","Right Ankle"),
 "left_upper_arm":("Left Shoulder","Left Elbow"),
 "right_upper_arm":("Right Shoulder","Right Elbow"),
 "left_forearm":("Left Elbow","Left Wrist"),
 "right_forearm":("Right Elbow","Right Wrist")
}
PAIRS=(
 ("pelvis_line","shoulder_line"),("left_upper_arm","right_upper_arm"),
 ("left_forearm","right_forearm"),("left_thigh","right_thigh"),
 ("left_shank","right_shank"),("trunk","left_thigh"),
 ("trunk","right_thigh"),("left_thigh","left_shank"),
 ("right_thigh","right_shank"),("left_upper_arm","right_thigh"),
 ("right_upper_arm","left_thigh")
)
assert len(PAIRS)==11
FRAME_COUNT=20
FPS=25.0

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def joint_points(doc):
    if doc.get("shooter_tracklet_id") is None:raise ValueError("Missing shooter tracklet")
    if not isinstance(doc.get("keypoint_3d"),dict):raise ValueError("No predicted 3D points")
    raw=doc["keypoint_3d"]
    if set(raw)!=set(str(i) for i in range(17)):
        raise ValueError("Must contain exactly original 17 annotated 3D joint keys")
    points=np.empty((17,3),dtype=np.float64)
    for name,index in JOINT_ORDER.items():
        field=raw[str(index)]
        if field["name"]!=name:raise ValueError(f"Joint order/name mismatch {index}")
        points[index]=[float(field[axis]) for axis in "xyz"]
    if not np.isfinite(points).all():raise ValueError("Nonfinite 3D keypoint")
    return points

def coordination(points):
    """20x17x3 original estimated coordinates -> 20x11 cosine relationships.

    Cosines cancel global translation, proper 3D rotation and positive uniform
    scale. They do NOT cancel camera-specific distortion, measurement error,
    morphological variation, or the left-right handedness of source skeleton.
    """
    p=np.asarray(points,dtype=float)
    if p.ndim!=3 or p.shape[1:]!=(17,3) or not np.isfinite(p).all():
        raise ValueError("Valid finite T x17x3 keypoints expected")
    vectors={}
    validity={}
    for name,(a,b) in SEGMENTS.items():
        diff=p[:,JOINT_ORDER[b]]-p[:,JOINT_ORDER[a]]
        length=np.linalg.norm(diff,axis=1)
        valid=length>1e-7
        unit=np.divide(diff,length[:,None],out=np.zeros_like(diff),where=valid[:,None])
        vectors[name]=unit
        validity[name]=valid
    values=np.column_stack([np.sum(vectors[a]*vectors[b],axis=1) for a,b in PAIRS])
    masks=np.column_stack([validity[a]&validity[b] for a,b in PAIRS])
    values=np.clip(values,-1.,1.)
    if np.any(~np.isfinite(values)) or np.any(np.abs(values[masks])>1.+1e-9):
        raise ValueError("Invalid coordination cosines")
    values[~masks]=np.nan
    return values,masks

def info_parser(raw):
    c=configparser.ConfigParser()
    c.read_string(raw.decode("utf-8-sig",errors="strict"))
    if set(c.sections())!={"info"}:raise ValueError("Missing original info section")
    meta={k:v for k,v in c["info"].items()}
    step=int(meta["position_step"])
    annotated=int(meta["annotated_position"])
    start=int(meta["start_position"])
    end=int(meta["end_position"])
    if step!=40 or not(0<=annotated-start<=FRAME_COUNT*step+100) or end<start:
        raise ValueError("Unexpected 3DSP sample clock")
    if meta["label"] not in ("Shots on target","Shots off target"):
        raise ValueError("Unexpected shot label")
    if not meta["game"] or not meta["id"]:raise ValueError("Missing source game ID")
    return meta, (np.arange(FRAME_COUNT)*step-(annotated-start))/1000.

def source_scan(z):
    names=set(z.namelist())
    clip_paths=sorted(n for n in names if re.fullmatch(r"3dsp/train/[0-9]{5}/info\.ini",n))
    if len(clip_paths)!=200:raise ValueError("Expected source version with exactly 200 training shot clips")
    # 10 official original test clips have no 3D postures; never convert
    # those to fake withheld labels and never mix them with a derived test.
    test_infos=[x for x in names if re.fullmatch(r"3dsp/test/[0-9]{5}/info\.ini",x)]
    if len(test_infos)!=10:raise ValueError("Expected 10 original 3DSP tracklet-only test clips")
    return clip_paths, len(test_infos)

def build(archive_path,output_dir,seed=3901):
    if sha(archive_path)!=SOURCE_SHA:raise ValueError("Source archive SHA256 differs from pinned upstream bytes")
    features,valid,clip_ids,games,labels,relative_times=[],[],[],[],[],[]
    anomalies=[]
    with zipfile.ZipFile(archive_path) as z:
        bad=z.testzip()
        if bad is not None:raise ValueError("Failed original ZIP CRC: "+bad)
        entries,official_test_count=source_scan(z)
        for info_path in entries:
            clip_id=info_path.split("/")[2]
            meta,times=info_parser(z.read(info_path))
            if meta["id"]!=clip_id:raise ValueError("Original clip ID differs from path")
            base=f"3dsp/train/{clip_id}/posture/"
            frames=[]
            errors=[]
            for t in range(1,FRAME_COUNT+1):
                f=f"{base}{t:03d}.json"
                try:
                    doc=json.loads(z.read(f))
                    frames.append(joint_points(doc))
                except (KeyError,ValueError,json.JSONDecodeError) as e:
                    errors.append({"frame":t,"error":str(e)})
            if errors:
                anomalies.append({"clip":clip_id,"reason":"Incomplete/malformed original 3D predicted frames",
                                  "errors":errors[:8]})
                continue
            xyz=np.stack(frames,axis=0)
            x,mask=coordination(xyz)
            if (~mask).any():
                anomalies.append({"clip":clip_id,"reason":"Degenerate 3D segment",
                                  "invalid_entries":int((~mask).sum())})
            features.append(x)
            valid.append(mask)
            clip_ids.append(clip_id)
            games.append(meta["game"])
            labels.append(meta["label"])
            relative_times.append(times)
    if len(features)<150:raise ValueError("Insufficient genuinely valid clips for this prespecified pilot")
    X=np.asarray(features,dtype=np.float32)
    M=np.asarray(valid,dtype=np.bool_)
    T=np.asarray(relative_times,dtype=np.float64)
    games=np.asarray(games,dtype="U160")
    labels=np.asarray(labels,dtype="U32")
    clips=np.asarray(clip_ids,dtype="U6")
    n_games=sorted(set(games.tolist()))
    if len(n_games)<10:raise ValueError("Cannot build a game-grouped holdout split")
    generator=random.Random(seed)
    generator.shuffle(n_games)
    val_groups=set(n_games[:4])
    test_groups=set(n_games[4:8])
    train_groups=set(n_games[8:])
    partition=np.asarray(["train" if g in train_groups else
                          "val" if g in val_groups else "test" for g in games])
    if min((partition==s).sum() for s in ("train","val","test"))<10:
        raise ValueError("Too few clips in a game-disjoint split")
    if train_groups&val_groups or train_groups&test_groups or val_groups&test_groups:
        raise ValueError("Game leakage")
    # Rare degenerate samples carry explicit masks. No leakage-imputation:
    # simply set invalid values to 0 and concatenate masks into model input.
    X_clean=np.where(M,X,0)
    if not np.isfinite(X_clean).all() or X_clean.shape[1:]!=(FRAME_COUNT,len(PAIRS)):
        raise ValueError("Invalid final features")
    output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/"coordination_3dsp_v039.npz"
    if output.exists():raise FileExistsError(output)
    np.savez_compressed(output,relation_cosines=X_clean,valid=M,
        time_relative_to_shot_ANNOTATION_s=T,clip_id=clips,
        source_game=games,source_label=labels,split=partition,
        relation_names=np.asarray([a+"__"+b for a,b in PAIRS]),
        verified_event_contact_truth=np.asarray(False),source_sha256=np.asarray(SOURCE_SHA))
    manifest=dict(
      schema_version=1,
      source="3DSP 2024, AutoSoccerPose, professional football broadcast source",
      original_commit=SOURCE_GIT,original_zip_sha256=SOURCE_SHA,
      original_archive_bytes=archive_path.stat().st_size,
      extracted_original_images=False,all_zip_crc_valid=True,
      source_train_clips=len(entries),source_test_clips_without_postures=official_test_count,
      accepted_clips=len(features),rejected_clips=len(anomalies),
      processed_frames=len(features)*FRAME_COUNT,relation_count=len(PAIRS),
      fps_assumed=FPS,shot_annotation_is_exact_foot_ball_contact=False,
      original_keypoint_type="MotionAGFormer video-lifted 3D (not optical MoCap or metrical biomechanics)",
      units_3d_calibrated=False,
      skilled_shot_source_context=True,
      verified_instep_contact=False,verified_strike_force=False,
      verified_ball_trajectory_or_power=False,
      verified_professional_explosive_3d_prior=False,
      license_boundary="Repository Apache-2.0 is not proof original SoccerNet cropped frames can be redistributed; extracted images and source ZIP remain ignored/local",
      data_output=str(output),data_output_sha256=sha(output),
      source_games_total=len(n_games),
      train_game_count=len(train_groups),val_game_count=len(val_groups),test_game_count=len(test_groups),
      splits={name:dict(n_clips=int((partition==name).sum()),
          game_names=sorted([g for g in n_games if g in {"train":train_groups,"val":val_groups,"test":test_groups}[name]]),
          on_target=int(((partition==name)&(labels=="Shots on target")).sum()),
          off_target=int(((partition==name)&(labels=="Shots off target")).sum()))
          for name in ("train","val","test")},
      keypoint_count=17,source_broadcast_annotation_event_time_frames_relative=T[0].tolist(),
      relation_names=[a+"__"+b for a,b in PAIRS],
      valid_fraction=float(M.mean()),missing_or_invalid_events=anomalies,
      guardrails=["Train/val/test are disjoint by MATCH, not just clip","All pose 3D is video-estimated, not a professional high-speed instep contact motion truth","Only 20 frames per clip, no known true ball impact time","No force plates, contact impulse, exact foot surface or biomechanics joint centers","Future projection model cannot control G1 without independent 3D IK and PhysX gate"],
      checksum_note="Data and original images remain untracked on Tang under data/ by project .gitignore."
    )
    dest=output_dir/"audit_manifest.json"
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print("V039_DATA_READY",json.dumps({k:manifest[k] for k in
       ("accepted_clips","rejected_clips","processed_frames","source_games_total","valid_fraction","data_output_sha256")}),
       flush=True)
    print("V039_SPLIT",json.dumps({k:v["n_clips"] for k,v in manifest["splits"].items()}),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="data/research_human_kicks/3dsp_cvpr2024/processed_v039")
    args=p.parse_args()
    build(ROOT/SOURCE_REL,ROOT/args.output_dir)

if __name__=="__main__":main()
