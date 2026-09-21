"""Project 2: Hough lane boundaries, SSD vehicles and conservative alerts."""
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def boundary_x(boundary, y):
    return boundary['a'] * y + boundary['b']


def marking_type(mask, a, b, y0, y1):
    """Classify observed paint runs; gaps are measured in image space."""
    h,w=mask.shape; ys=np.arange(max(0,int(y0)),min(h,int(y1)+1)); xs=np.rint(a*ys+b).astype(int)
    valid=(xs>=4)&(xs<w-4);ys,xs=ys[valid],xs[valid]
    if len(ys)<30:return 'unknown',0.
    hits=np.array([np.any(mask[y,x-4:x+5]) for x,y in zip(xs,ys)],np.uint8)
    hits=cv2.morphologyEx(hits[:,None],cv2.MORPH_CLOSE,np.ones((5,1),np.uint8)).ravel()
    occupied=np.flatnonzero(hits)
    if len(occupied)<10:return 'unknown',float(hits.mean())
    core=hits[occupied[0]:occupied[-1]+1]; changes=np.diff(np.r_[1,core,1].astype(int))
    starts=np.flatnonzero(changes==-1);ends=np.flatnonzero(changes==1)
    substantial=sum((ends-starts)>=max(5,.018*h));coverage=float(core.mean())
    if substantial>=1 and coverage<.85:return 'dashed',coverage
    if coverage>=.85 and len(core)>.12*h:return 'solid',coverage
    return 'unknown',coverage


def detect_boundaries(image):
    h,w=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    white=cv2.inRange(hsv,np.array([0,0,145]),np.array([179,85,255]))
    yellow=cv2.inRange(hsv,np.array([15,75,100]),np.array([40,255,255]))
    paint=cv2.bitwise_or(white,yellow)
    roi=np.zeros_like(gray)
    cv2.fillPoly(roi,[np.int32([[0,h-1],[.40*w,.43*h],[.63*w,.43*h],[w-1,h-1]])],255)
    paint=cv2.bitwise_and(paint,roi)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
    edges=cv2.bitwise_and(edges,cv2.dilate(paint,np.ones((3,3),np.uint8)))
    segments=cv2.HoughLinesP(edges,1,np.pi/180,threshold=20,minLineLength=max(15,int(.035*h)),maxLineGap=max(8,int(.035*h)))
    records=[]
    if segments is not None:
        for x1,y1,x2,y2 in segments[:,0]:
            if abs(y2-y1)<max(12,.025*h):continue
            a=(x2-x1)/(y2-y1);b=x1-a*y1;bottom=a*(h-1)+b;top=a*(.48*h)+b
            if abs(a)>2.8 or not -.35*w<bottom<1.35*w or not .15*w<top<.85*w:continue
            records.append((a,b,float(np.hypot(x2-x1,y2-y1)),(x1,y1,x2,y2)))
    if not records:return [],paint
    # Cluster repeated Hough detections by extrapolated road intercept and slope.
    descriptors=np.array([[(a*(h-1)+b)/(.035*w),a/.20] for a,b,_,_ in records])
    labels=DBSCAN(eps=1,min_samples=1).fit_predict(descriptors);boundaries=[]
    for label in sorted(set(labels)):
        group=[r for r,l in zip(records,labels) if l==label]
        points=np.array([pt for r in group for pt in [(r[3][0],r[3][1]),(r[3][2],r[3][3])]],float)
        ys=points[:,1];xs=points[:,0]
        if np.ptp(ys)<.06*h:continue
        a,b=np.polyfit(ys,xs,1);kind,coverage=marking_type(paint,a,b,max(.5*h,ys.min()),h-1)
        boundaries.append(dict(a=float(a),b=float(b),kind=kind,paint_coverage=coverage,
            support=float(sum(r[2] for r in group)),y_start=int(max(.47*h,ys.min())),y_end=int(h-1)))
    # Select the dominant vanishing point from pairwise Hough-line intersections.
    intersections=[];weights=[]
    for i,left in enumerate(boundaries):
        for right in boundaries[i+1:]:
            delta=left['a']-right['a']
            if abs(delta)<.15:continue
            y=(right['b']-left['b'])/delta;x=boundary_x(left,y)
            if .28*h<y<.72*h and .25*w<x<.85*w:
                intersections.append((x,y));weights.append(np.sqrt(left['support']*right['support']))
    if intersections:
        points=np.array(intersections);weights=np.array(weights)
        distances=np.linalg.norm((points[:,None]-points[None,:])/np.array([.045*w,.045*h]),axis=2)
        best=np.argmax((distances<1)@weights);near=distances[best]<1
        vx,vy=np.average(points[near],axis=0,weights=weights[near])
        boundaries=[b for b in boundaries if abs(boundary_x(b,vy)-vx)<.055*w
                    and -.1*w<boundary_x(b,h-1)<1.1*w
                    and b['y_start']<.9*h]
    boundaries.sort(key=lambda l:boundary_x(l,h-1))
    return boundaries,paint


def make_lanes(boundaries,shape):
    h,w=shape[:2];lanes=[]
    for left,right in zip(boundaries,boundaries[1:]):
        width=boundary_x(right,h-1)-boundary_x(left,h-1)
        if .07*w<width<.8*w and boundary_x(right,.65*h)>boundary_x(left,.65*h):
            lanes.append({'id':len(lanes)+1,'left':left,'right':right})
    return lanes


def assign_lane(x,y,lanes):
    for lane in lanes:
        if boundary_x(lane['left'],y)<=x<=boundary_x(lane['right'],y):return lane['id']
    return None


class VehicleDetector:
    def __init__(self,weights):
        weights=Path(weights)
        self.net=cv2.dnn.readNetFromCaffe(str(weights/'deploy.prototxt'),str(weights/'mobilenet_iter_73000.caffemodel'))

    def detect_single(self,image,threshold=.3):
        h,w=image.shape[:2];self.net.setInput(cv2.dnn.blobFromImage(image,.007843,(300,300),127.5))
        output=self.net.forward();boxes=[];scores=[];classes=[]
        for row in output[0,0]:
            cls=int(row[1]);score=float(row[2])
            if cls not in (6,7) or score<threshold:continue
            x1,y1,x2,y2=(row[3:7]*[w,h,w,h]).astype(int)
            x1,x2=np.clip([x1,x2],0,w-1);y1,y2=np.clip([y1,y2],0,h-1)
            if x2<=x1 or y2<=y1:continue
            boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1)]);scores.append(score);classes.append(cls)
        keep=cv2.dnn.NMSBoxes(boxes,scores,threshold,.45)
        return [{'box':boxes[int(i)],'confidence':scores[int(i)],'class':'car' if classes[int(i)]==7 else 'bus'} for i in np.asarray(keep).ravel()]

    def detect(self,image,threshold=.3):
        h,w=image.shape[:2];detections=self.detect_single(image,threshold)
        # Overlapping crops preserve small cars that disappear in 300px resizing.
        for x0,y0 in [(0,int(.3*h)),(int(.35*w),int(.3*h)),(int(.175*w),int(.2*h))]:
            crop=image[y0:min(h,y0+int(.65*h)),x0:min(w,x0+int(.65*w))]
            for d in self.detect_single(crop,threshold):
                d['box'][0]+=x0;d['box'][1]+=y0;detections.append(d)
        boxes=[d['box'] for d in detections];scores=[d['confidence'] for d in detections]
        keep=cv2.dnn.NMSBoxes(boxes,scores,threshold,.4)
        return [detections[int(i)] for i in np.asarray(keep).ravel()]


class CrossingTracker:
    """A temporal image-space crossing cue, assuming a fixed forward-facing camera."""
    def __init__(self):self.previous=[];self.cooldown=0
    def update(self,boundaries,shape):
        h,w=shape[:2];xs=[boundary_x(b,.95*h)/w for b in boundaries]
        event=False;updated=[];available=list(self.previous);self.cooldown=max(0,self.cooldown-1)
        for raw in xs:
            match=min(available,key=lambda item:abs(item['x']-raw)) if available else None
            if match is not None and abs(match['x']-raw)<.10:
                available.remove(match);x=.5*raw+.5*match['x'];stable=match['side'];count=match['count']
            else:
                x=raw;stable=-1 if x<.48 else 1 if x>.52 else 0;count=0
            side=-1 if x<.48 else 1 if x>.52 else 0
            if side and stable and side!=stable:
                count+=1
                if count>=3:
                    if self.cooldown==0:event=True;self.cooldown=20
                    stable=side;count=0
            elif side:stable=side;count=0
            else:count=0
            updated.append(dict(x=x,side=stable,count=count))
        self.previous=updated
        return event


def analyze(image,detector=None,tracker=None,focal_px=None,vehicle_height_m=1.5):
    h,w=image.shape[:2];boundaries,paint=detect_boundaries(image);lanes=make_lanes(boundaries,image.shape)
    ego=assign_lane(w/2,.95*h,lanes);vehicles=detector.detect(image) if detector else []
    for v in vehicles:
        x,y,bw,bh=v['box'];v['lane_id']=assign_lane(x+bw/2,y+bh,lanes)
        v['image_height_fraction']=bh/h
        v['close_proxy']=bool(bh/h>.18)
        v['close_vehicle_alert']=v['close_proxy']
        v['distance_m_estimate']=float(focal_px*vehicle_height_m/bh) if focal_px is not None else None
        v['ahead_close_alert']=bool(v['close_proxy'] and ego is not None and v['lane_id']==ego)
    near=bool(ego is not None and any(abs(boundary_x(b,.95*h)-w/2)<.055*w for b in boundaries))
    result={'boundaries':boundaries,'lanes':lanes,'ego_lane_id':ego,'vehicles':vehicles,
            'near_boundary_alert':near,'possible_crossing':tracker.update(boundaries,image.shape) if tracker else None,
            'camera_assumption':'centered, forward-facing; ego lane is unavailable for elevated roadside views',
            'distance_note':'uncalibrated height proxy' if focal_px is None else 'pinhole estimate using assumed vehicle height'}
    overlay=image.copy()
    for lane in lanes:
        yy=[int(.6*h),h-1];poly=np.int32([[boundary_x(lane['left'],y),y] for y in yy]+[[boundary_x(lane['right'],y),y] for y in yy[::-1]])
        cv2.fillPoly(overlay,[poly],(30,140,30) if lane['id']==ego else (120,60,20))
    annotated=cv2.addWeighted(overlay,.22,image,.78,0)
    for b in boundaries:
        pts=tuple((int(boundary_x(b,y)),int(y)) for y in [b['y_start'],h-1])
        cv2.line(annotated,*pts,(0,220,255),2)
        tx=int(np.clip(boundary_x(b,.8*h),0,max(0,w-100)));cv2.putText(annotated,b['kind'],(tx,int(.8*h)),0,.5,(0,255,255),1)
    for v in vehicles:
        x,y,bw,bh=v['box'];color=(0,0,255) if v['ahead_close_alert'] else (255,180,0)
        cv2.rectangle(annotated,(x,y),(x+bw,y+bh),color,2)
        cv2.putText(annotated,f"{v['class']} lane {v['lane_id']} {v['confidence']:.2f}",(x,max(15,y-5)),0,.45,color,1)
    cv2.putText(annotated,f"ego lane: {ego} | boundary proximity: {near}",(10,25),0,.55,(0,0,255),2)
    return result,annotated,paint


def run_images(folder,outdir,weights):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);detector=VehicleDetector(weights);results={}
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in ('.jpg','.png','.jpeg'):continue
        im=cv2.imread(str(path));scale=min(1,1280/max(im.shape[:2]));im=cv2.resize(im,None,fx=scale,fy=scale)
        record,annotated,paint=analyze(im,detector)
        # road2 and road4 are elevated roadside cameras, not ego camera views.
        if path.stem.startswith(('road2','road4')):
            record['ego_lane_id']=None;record['near_boundary_alert']=None
            for v in record['vehicles']:v['ahead_close_alert']=False
            cv2.rectangle(annotated,(0,0),(annotated.shape[1],35),(0,0,0),-1)
            cv2.putText(annotated,'Roadside camera: ego lane unavailable',(10,25),0,.55,(255,255,255),1)
        results[path.name]=record;cv2.imwrite(str(out/f'{path.stem}_annotated.jpg'),annotated)
        cv2.imwrite(str(out/f'{path.stem}_paint.png'),paint)
    (out/'detections.json').write_text(json.dumps(results,indent=2));return results


def run_video(path,outdir,weights,max_frames=300):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot read video: {path}')
    detector=VehicleDetector(weights);tracker=CrossingTracker();records=[];writer=None
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    try:
        for i in range(max_frames):
            ok,frame=cap.read()
            if not ok:break
            scale=min(1,960/max(frame.shape[:2]));frame=cv2.resize(frame,None,fx=scale,fy=scale)
            record,annotated,_=analyze(frame,detector,tracker)
            records.append(dict(frame=i,time_seconds=i/fps,**record))
            if writer is None:
                writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():raise RuntimeError('MP4 video writer unavailable')
            writer.write(annotated)
    finally:
        cap.release()
        if writer:writer.release()
    (out/'frames.json').write_text(json.dumps(records,indent=2));return len(records)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--weights',default='data/weights');p.add_argument('--out',default='results/project2');p.add_argument('--video',action='store_true');p.add_argument('--max-frames',type=int,default=300)
    a=p.parse_args()
    if a.video:print(run_video(a.input,a.out,a.weights,a.max_frames))
    else:
        r=run_images(a.input,a.out,a.weights)
        print(json.dumps({k:{'boundaries':len(v['boundaries']),'vehicles':len(v['vehicles']),'ego_lane':v['ego_lane_id']} for k,v in r.items()},indent=2))
