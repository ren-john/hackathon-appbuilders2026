"""Local infant pose + toy detection. R: infant ROI, A: full frame, Q: quit.
MediaPipe primary; FiDIP fallback. Select one infant in the ROI.
Pose and 2D proximity are estimates, including when the infant is covered.
"""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import mediapipe as mp
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from ultralytics import YOLO
from test2_fidip import FiDIP

HERE = Path(__file__).resolve().parent
TORSO = [5, 6, 11, 12]
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def valid_points(points, scores, threshold):
    return (np.isfinite(points).all(axis=1) & np.isfinite(scores) & (scores >= threshold)
            & (points >= 0).all(axis=1) & (points <= 1).all(axis=1))


def body_bounds(points, valid):
    if not valid[TORSO].all():
        return None
    low, high = points[valid].min(axis=0), points[valid].max(axis=0)
    return [*low.tolist(), *high.tolist()] if (high > low).all() else None


def near_body(box, baby, margin):
    if baby is None:
        return None
    dx = max(baby[0]-box[2], box[0]-baby[2], 0)
    dy = max(baby[1]-box[3], box[1]-baby[3], 0)
    return bool(math.hypot(dx, dy) <= margin * math.hypot(baby[2]-baby[0], baby[3]-baby[1]))


def covers_body(box, baby, points, valid):
    """Optional heuristic for a whole-body false toy detection, not a semantic test."""
    if baby is None or valid.sum() < 8:
        return False
    inside = ((points[:,0] >= box[0]) & (points[:,0] <= box[2]) &
              (points[:,1] >= box[1]) & (points[:,1] <= box[3]))
    area = (baby[2]-baby[0])*(baby[3]-baby[1])
    ratio = (box[2]-box[0])*(box[3]-box[1])/area
    return bool(inside[TORSO].all() and inside[valid].mean() >= .85 and .75 <= ratio <= 1.8)


class BabyPose:
    def __init__(self, args):
        self.args = args
        self.fallback = None
        self.recovery = 0
        self.active = False
        self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.pose_model)),
            running_mode=RunningMode.VIDEO, num_poses=1))

    def reset(self):
        self.active, self.recovery = False, 0

    def close(self):
        self.pose.close()

    def process(self, frame, roi, timestamp):
        h, w = frame.shape[:2]
        x0,y0,x1,y1 = roi or (0,0,w,h)
        x0,y0,x1,y1 = max(0,x0),max(0,y0),min(w,x1),min(h,y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Infant ROI does not intersect the image')
        rgb = cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
        result = self.pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
        points, scores = np.zeros((17,2)), np.zeros(17)
        if result.pose_landmarks:
            lm = [result.pose_landmarks[0][i] for i in MP_TO_COCO]
            points = np.array([[p.x,p.y] for p in lm])
            scores = np.array([min(p.visibility or 0,p.presence or 0) for p in lm])
        valid = valid_points(points,scores,self.args.mp_threshold)
        ready = bool(valid[TORSO].all() and valid.sum() >= 6)
        if not ready:
            self.active, self.recovery = True, 0
        elif self.active:
            self.recovery += 1
            if self.recovery >= self.args.recovery_frames:
                self.active, self.recovery = False, 0
        source = 'MediaPipe'
        if self.active or self.args.force_fallback:
            if self.fallback is None:
                self.fallback = FiDIP(SimpleNamespace(
                    repo=HERE/'Infant-Pose-Estimation', config=None, model=self.args.fidip_model,
                    device=('cuda' if self.args.device=='0' else self.args.device), flip_test=False))
            points,scores = self.fallback.predict(rgb)
            valid = valid_points(points,scores,self.args.fidip_threshold)
            source = 'FiDIP fallback'
        pixels = points*np.array([x1-x0,y1-y0])+[x0,y0]
        baby = body_bounds(pixels,valid)
        return dict(source=source, baby_box=baby, roi=[x0,y0,x1,y1]), pixels, valid


def process_frame(frame, pose, detector, args, roi, timestamp):
    info, points, valid = pose.process(frame,roi,timestamp)
    result = detector.predict(frame,conf=args.threshold,device=args.device,
                              imgsz=args.imgsz,verbose=False)[0]
    keep, objects, suppressed = [], [], []
    for i, box in enumerate(result.boxes.cpu()):
        label = result.names[int(box.cls.item())]
        bounds = box.xyxy[0].tolist()
        if label.lower() in {'person','baby','infant'}:
            suppressed.append(dict(label=label, reason='person class'))
            continue
        if args.suppress_body_overlap and label in {'hard-toy','soft-toy'} and covers_body(bounds,info['baby_box'],points,valid):
            suppressed.append(dict(label=label, reason='whole-body overlap heuristic'))
            continue
        near = near_body(bounds,info['baby_box'],args.near_margin)
        objects.append(dict(label=label,score=round(float(box.conf.item()),3),box=bounds,near_baby=near))
        keep.append(i)
    view = result[keep].plot() if keep else frame.copy()
    for a,b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(view,tuple(points[a].astype(int)),tuple(points[b].astype(int)),(0,220,0),2)
    for p in points[valid]:
        cv2.circle(view,tuple(p.astype(int)),3,(0,70,255),-1)
    x0,y0,x1,y1=info['roi']
    cv2.rectangle(view,(x0,y0),(x1-1,y1-1),(255,180,0),1)
    messages=sorted(set(f"{o['label']} detected near baby" for o in objects if o['near_baby']))
    status = ('Baby location unknown; proximity unavailable' if info['baby_box'] is None else
              ' | '.join(messages) if messages else 'No recognized object detected near baby')
    # Header stays outside the image so detections are not hidden by status text.
    import textwrap
    lines=[f"Pose: {info['source']} | Objects: {len(objects)} | Suppressed: {len(suppressed)}"]
    lines+=textwrap.wrap(status,width=max(40,frame.shape[1]//10))
    header=np.zeros((28*len(lines)+12,max(frame.shape[1],640),3),np.uint8)
    for i,line in enumerate(lines):
        cv2.putText(header,line,(8,24+28*i),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    canvas=np.zeros((header.shape[0]+view.shape[0],header.shape[1],3),np.uint8)
    canvas[:header.shape[0]]=header
    canvas[header.shape[0]:,:view.shape[1]]=view
    info.update(objects=objects,suppressed=suppressed,status=status)
    return canvas,info


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='0',help='Camera index or video')
    p.add_argument('--image',type=Path,help='Test one still image instead')
    p.add_argument('--model',type=Path,default=HERE/'Models/cribhd_t_best_20261009.pt')
    p.add_argument('--pose-model',type=Path,default=HERE/'Models/pose_landmarker_lite.task')
    p.add_argument('--fidip-model',type=Path,default=HERE/'Models/hrnet_fidip.pth')
    p.add_argument('--threshold',type=float,default=.65)
    p.add_argument('--mp-threshold',type=float,default=.5)
    p.add_argument('--fidip-threshold',type=float,default=.2)
    p.add_argument('--near-margin',type=float,default=.25)
    p.add_argument('--recovery-frames',type=int,default=5)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--device',choices=['auto','cpu','0'],default='auto')
    p.add_argument('--box',type=int,nargs=4,metavar=('X0','Y0','X1','Y1'))
    p.add_argument('--suppress-body-overlap',action='store_true',help='Heuristic: hide toy boxes covering almost all of the pose; can hide real objects')
    p.add_argument('--force-fallback',action='store_true',help='Diagnostic: force FiDIP')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args()
    for name in ('threshold','mp_threshold','fidip_threshold'):
        if not math.isfinite(getattr(a,name)) or not 0<=getattr(a,name)<=1:p.error(f'{name} must be in [0,1]')
    if not math.isfinite(a.near_margin) or a.near_margin<0:p.error('Invalid near margin')
    if a.recovery_frames<1 or a.imgsz<32 or a.max_frames<0:p.error('Invalid frame count/image size')
    if a.box and (a.box[2]<=a.box[0] or a.box[3]<=a.box[1]):p.error('Invalid ROI')
    for path in (a.model,a.pose_model,a.fidip_model):
        if not path.is_file():p.error(f'Missing model: {path}')
    a.device=('0' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    return a


def main():
    args=parse_args()
    detector=YOLO(str(args.model))
    if detector.task not in ('detect','segment'):raise ValueError('YOLO model must detect or segment')
    print(f'YOLO: {args.model} | classes: {detector.names} | device: {args.device}',flush=True)
    pose=BabyPose(args)
    capture=None
    try:
        if args.image:
            from PIL import Image, ImageOps
            with Image.open(args.image) as im:frame=cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert('RGB')),cv2.COLOR_RGB2BGR)
            view,info=process_frame(frame,pose,detector,args,args.box,0)
            from datetime import datetime
            import uuid
            out=HERE/'test2_results'/f'{args.image.stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
            out.mkdir(parents=True,exist_ok=False)
            if not cv2.imwrite(str(out/'prediction.png'),view):raise IOError('Image save failed')
            (out/'report.json').write_text(json.dumps(info,indent=2))
            print(json.dumps(info),flush=True); print(f'Saved: {out}',flush=True)
            return
        camera=args.source.isdecimal()
        capture=cv2.VideoCapture(int(args.source) if camera else args.source)
        if not capture.isOpened():raise RuntimeError(f'Cannot open {args.source}')
        fps=capture.get(cv2.CAP_PROP_FPS)
        fps=fps if math.isfinite(fps) and fps>0 else 30
        origin=time.monotonic();last=-1;count=0;roi=args.box;previous=None
        title='Baby pose + YOLO toys | R: ROI, A: full frame, Q: quit'
        while True:
            ok,frame=capture.read()
            if not ok:
                if not count:raise RuntimeError('No frames from source')
                break
            timestamp=max(last+1,int((time.monotonic()-origin)*1000 if camera else count*1000/fps));last=timestamp
            view,info=process_frame(frame,pose,detector,args,roi,timestamp)
            signature=(info['source'],info['status'])
            if signature!=previous or count%30==0:print(json.dumps(dict(frame=count,**info)),flush=True)
            previous=signature;count+=1
            if not args.headless:
                cv2.imshow(title,view)
                key=cv2.waitKey(1)&0xff
                if key in (ord('q'),27) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
                if key==ord('r'):
                    x,y,w,h=cv2.selectROI('Select one infant',frame,showCrosshair=True)
                    cv2.destroyWindow('Select one infant')
                    if w>1 and h>1:roi=[x,y,x+w,y+h];pose.reset()
                elif key==ord('a'):roi=None;pose.reset()
            if args.max_frames and count>=args.max_frames:break
        print(f'Processed {count} frames')
    finally:
        pose.close()
        if capture is not None:capture.release()
        if not args.headless:cv2.destroyAllWindows()


if __name__=='__main__':
    main()
