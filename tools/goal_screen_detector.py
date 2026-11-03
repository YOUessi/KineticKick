"""Research-only RGB screenshot goal-post detector.

This uses rendered Isaac Sim viewport pixels, NOT USD object coordinates, and
is NOT yet mounted on G1's RGB camera. It measures visible blue goal post pixel
location and draws a box; 3D goal pose remains from USD until onboard vision.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label,find_objects

def detect_goal_rgb(view):
    rgb=np.asarray(view).astype(np.int16)
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    h,w=r.shape
    # Crop viewport, not the side panels: default screenshot 1280x798.
    xx=np.arange(w)[None,:]
    yy=np.arange(h)[:,None]
    in_view=(xx>w*.05)&(xx<w*.70)&(yy>h*.075)&(yy<h*.66)
    mask=(b>140)&(b-g>22)&(b-r>55)&(r<155)&in_view
    connected,count=label(mask)
    props=[]
    for i,sl in enumerate(find_objects(connected),start=1):
        if sl is None:continue
        ys,xs=sl
        area=int((connected[sl]==i).sum())
        rw=xs.stop-xs.start
        rh=ys.stop-ys.start
        if area>=40 and rh>=18 and rw<=145:
            props.append((area,[xs.start,ys.start,xs.stop,ys.stop]))
    props.sort(reverse=True)
    if not props:return None,mask
    box=props[0][1]
    return {"bbox_px":box,"center_px":[round((box[0]+box[2])/2,1),round((box[1]+box[3])/2,1)],
            "blue_area":props[0][0], "candidates":len(props),
            "method":"screen RGB blue-post segment, not robot camera"},mask

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    img=Image.open(args.image).convert("RGB")
    out,mask=detect_goal_rgb(img)
    canvas=img.copy()
    draw=ImageDraw.Draw(canvas)
    if out:
        draw.rectangle(out["bbox_px"],outline=(255,40,36),width=4)
        draw.text((out["bbox_px"][0],max(5,out["bbox_px"][1]-17)), "RGB GOAL POST", fill=(255,40,36))
    Path(args.output).parent.mkdir(exist_ok=True,parents=True)
    canvas.save(args.output)
    print("GOAL_RGB_SCREEN",json.dumps(out),flush=True)
if __name__=="__main__":
    main()
