"""Project 1: classical segmentation, engineered descriptors and grouped classification."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern


def read_rgb(path,size=256):
    image=cv2.imread(str(path))
    if image is None:raise ValueError(f'Unreadable image: {path}')
    h,w=image.shape[:2]
    # Preserve aspect ratio; features use relative rather than clinical millimeter size.
    scale=size/max(h,w)
    image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def segment(image):
    """Otsu color-distance segmentation with connected-component selection."""
    rgb=np.asarray(image,dtype=np.uint8);h,w=rgb.shape[:2]
    lab=cv2.cvtColor(cv2.GaussianBlur(rgb,(5,5),0),cv2.COLOR_RGB2LAB).astype(float)
    yy,xx=np.mgrid[:h,:w];relative=np.sqrt(((xx-w/2)/(w/2))**2+((yy-h/2)/(h/2))**2)
    peripheral=(relative>.7)&(relative<1.)&(lab[:,:,0]>40)
    background=np.median(lab[peripheral],axis=0) if peripheral.any() else np.median(lab.reshape(-1,3),axis=0)
    distance=np.sqrt(np.sum(((lab-background)*np.array([.7,1.,1.]))**2,axis=2))
    distance=np.uint8(np.clip(distance,0,255))
    field=(relative<1.15)&(lab[:,:,0]>40)
    field=cv2.erode(field.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    if not field.any():raise ValueError('No usable illuminated field')
    threshold,_=cv2.threshold(distance[field],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary=np.uint8((distance>threshold)&field)*255
    binary[:3]=0;binary[-3:]=0;binary[:,:3]=0;binary[:,-3:]=0
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(binary)
    candidates=[]
    for k in range(1,count):
        area=stats[k,cv2.CC_STAT_AREA]
        if area<max(10,.001*h*w):continue
        center_distance=np.linalg.norm((centroids[k]-[w/2,h/2])/[w,h])
        candidates.append((area/(1+4*center_distance),k))
    if not candidates:raise ValueError('Segmentation failed: no substantial foreground component')
    selected=(labels==max(candidates)[1]).astype(np.uint8)
    contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(selected);cv2.drawContours(mask,contours,-1,1,thickness=cv2.FILLED)
    return mask.astype(bool)


def preprocess(image):
    smooth=cv2.bilateralFilter(image,5,35,35)
    gray=cv2.cvtColor(smooth,cv2.COLOR_RGB2GRAY)
    blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    hairs=np.uint8(blackhat>25)*255
    smooth=cv2.inpaint(smooth,hairs,3,cv2.INPAINT_TELEA)
    lab=cv2.cvtColor(smooth,cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    return smooth,enhanced


def shape_features(mask):
    points=np.column_stack(np.where(mask)[::-1]).astype(float);area=len(points)
    if area<10:raise ValueError('Mask too small for reliable features')
    contour=max(cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
    perimeter=cv2.arcLength(contour,True);contour_area=cv2.contourArea(contour)
    hull_area=cv2.contourArea(cv2.convexHull(contour))
    eig,axes=np.linalg.eigh(np.cov(points.T));angle=np.degrees(np.arctan2(axes[1,-1],axes[0,-1]))
    h,w=mask.shape;center=points.mean(axis=0);canvas=max(h,w)*2
    transform=cv2.getRotationMatrix2D(tuple(center),float(angle),1)
    transform[:,2]+=np.array([canvas/2,canvas/2])-center
    aligned=cv2.warpAffine(mask.astype(np.uint8),transform,(canvas,canvas),flags=cv2.INTER_NEAREST)
    ys,xs=np.where(aligned);aligned=aligned[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(bool)
    asym_x=np.logical_xor(aligned,aligned[:,::-1]).sum()/(2*aligned.sum())
    asym_y=np.logical_xor(aligned,aligned[::-1]).sum()/(2*aligned.sum())
    x,y,bw,bh=cv2.boundingRect(contour)
    return dict(area_fraction=area/(h*w),equivalent_diameter_relative=np.sqrt(4*area/np.pi)/np.hypot(h,w),
                circularity=4*np.pi*contour_area/(perimeter**2+1e-9),
                boundary_irregularity=perimeter/(2*np.sqrt(np.pi*max(contour_area,1))),
                solidity=contour_area/max(hull_area,1),extent=area/(bw*bh),
                eccentricity=np.sqrt(max(0,1-eig[0]/max(eig[1],1e-8))),asymmetry_major=asym_x,asymmetry_minor=asym_y)


def masked_texture(gray,mask):
    levels=16;q=np.minimum(gray//16,15).astype(int);values=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1),(0,3),(3,0)]:
        h,w=gray.shape;y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
        a=q[y0:y1,x0:x1];b=q[y0+dy:y1+dy,x0+dx:x1+dx]
        valid=mask[y0:y1,x0:x1]&mask[y0+dy:y1+dy,x0+dx:x1+dx]
        counts=np.bincount(a[valid]*levels+b[valid],minlength=levels**2).reshape(levels,levels).astype(float)
        counts+=counts.T.copy();p=counts/max(counts.sum(),1)
        i,j=np.indices(p.shape);nz=p[p>0]
        values.append([np.sum(p*(i-j)**2)/225,np.sum(p/(1+np.abs(i-j))),np.sum(p*p),-np.sum(nz*np.log2(nz))])
    return dict(zip(['texture_contrast','texture_homogeneity','texture_energy','texture_entropy'],np.mean(values,axis=0)))


def extract_features(path):
    image=read_rgb(path);mask=segment(image);smooth,enhanced=preprocess(image)
    result=shape_features(mask)
    for space,array in [('rgb',smooth),('lab',cv2.cvtColor(smooth,cv2.COLOR_RGB2LAB)),('hsv',cv2.cvtColor(smooth,cv2.COLOR_RGB2HSV))]:
        for channel in range(3):
            values=array[:,:,channel][mask].astype(float)/255
            for label,value in [('mean',values.mean()),('std',values.std()),('p10',np.percentile(values,10)),('p90',np.percentile(values,90))]:
                result[f'{space}_{channel}_{label}']=float(value)
    gray=cv2.cvtColor(enhanced,cv2.COLOR_RGB2GRAY)
    lbp=local_binary_pattern(gray,8,1,method='uniform')
    inner=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if not inner.any():inner=mask
    hist=np.bincount(lbp[inner].astype(int),minlength=10).astype(float);hist/=hist.sum()
    result.update({f'lbp_{i}':float(v) for i,v in enumerate(hist)})
    result.update(masked_texture(gray,mask))
    edges=cv2.Canny(gray,60,140)>0;result['edge_density']=float(edges[inner].mean())
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite feature')
    return result


def build_features(manifest,outfile,workers=4):
    from concurrent.futures import ThreadPoolExecutor
    manifest=Path(manifest);data=pd.read_csv(manifest);root=manifest.parent
    cv2.setNumThreads(1)
    def worker(row):
        return dict(image=row.image,label=int(row.label),group=str(row.group),**extract_features(root/row.path))
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(worker,data.itertuples(index=False)))
    result=pd.DataFrame(rows);Path(outfile).parent.mkdir(parents=True,exist_ok=True);result.to_csv(outfile,index=False)
    return result


def train(features,outdir,seed=42):
    from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report,confusion_matrix,balanced_accuracy_score,roc_auc_score,
                                 average_precision_score,accuracy_score,f1_score,RocCurveDisplay,ConfusionMatrixDisplay)
    import joblib,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.read_csv(features);out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    names=[c for c in frame if c not in ['image','label','group']]
    X,y,groups=frame[names].to_numpy(),frame.label.to_numpy(),frame.group.astype(str).to_numpy()
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    tr,te=next(outer.split(X,y,groups));assert not set(groups[tr])&set(groups[te])
    if len(np.unique(y[tr]))!=2 or len(np.unique(y[te]))!=2:raise ValueError('Both classes must occur in train and test')
    splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(X[tr],y[tr],groups[tr]))
    for a,b in splits:assert not set(groups[tr][a])&set(groups[tr][b])
    experiments={
      'svm':(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),SVC(class_weight='balanced')),
             {'svc__C':[.1,1,10],'svc__gamma':['scale',.01,.1]}),
      'random_forest':(make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=seed,n_jobs=1)),
                       {'randomforestclassifier__max_depth':[8,None],'randomforestclassifier__min_samples_leaf':[2,5]})}
    fits={};cv_scores={}
    for name,(estimator,grid) in experiments.items():
        search=GridSearchCV(estimator,grid,cv=splits,scoring='balanced_accuracy',n_jobs=2).fit(X[tr],y[tr])
        fits[name]=search;cv_scores[name]={'balanced_accuracy':float(search.best_score_),'parameters':search.best_params_}
    best=max(fits,key=lambda k:fits[k].best_score_);model=fits[best].best_estimator_
    prediction=model.predict(X[te]);score=model.decision_function(X[te]) if hasattr(model,'decision_function') else model.predict_proba(X[te])[:,1]
    cm=confusion_matrix(y[te],prediction,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    result={'sample_count':len(frame),'train_count':len(tr),'test_count':len(te),'feature_count':len(names),'seed':seed,
            'train_class_counts':np.bincount(y[tr],minlength=2).tolist(),'test_class_counts':np.bincount(y[te],minlength=2).tolist(),
            'selected_model':best,'cross_validation':cv_scores,'accuracy':accuracy_score(y[te],prediction),
            'balanced_accuracy':balanced_accuracy_score(y[te],prediction),'melanoma_f1':f1_score(y[te],prediction),
            'roc_auc':roc_auc_score(y[te],score),'average_precision':average_precision_score(y[te],score),
            'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),'confusion_matrix':cm.tolist(),
            'classification_report':classification_report(y[te],prediction,target_names=['nevus','melanoma'],output_dict=True),
            'scope':'Balanced ISIC-2019 subset; lesion-disjoint splits where lesion_id is known; missing IDs are image-specific groups. Not patient-disjoint or a full-dataset benchmark.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    split=frame[['image','label','group']].copy();split['split']='train';split.loc[te,'split']='test';split.to_csv(out/'split_manifest.csv',index=False)
    pd.DataFrame({'image':frame.image.iloc[te],'label':y[te],'prediction':prediction,'score':score}).to_csv(out/'test_predictions.csv',index=False)
    joblib.dump({'model':model,'feature_names':names,'class_mapping':{0:'nevus',1:'melanoma'}},out/'classifier.joblib')
    fig,axes=plt.subplots(1,2,figsize=(10,4));ConfusionMatrixDisplay(cm,display_labels=['NV','MEL']).plot(ax=axes[0],colorbar=False)
    RocCurveDisplay.from_predictions(y[te],score,ax=axes[1]);fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=140);plt.close(fig)
    return result


def predict(image,checkpoint):
    import joblib
    saved=joblib.load(checkpoint);features=extract_features(image)
    vector=np.array([[features[k] for k in saved['feature_names']]])
    return saved['class_mapping'][int(saved['model'].predict(vector)[0])]

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('features');f.add_argument('manifest');f.add_argument('--out',default='results/project1/features.csv')
    t=sub.add_parser('train');t.add_argument('features');t.add_argument('--out',default='results/project1')
    q=sub.add_parser('predict');q.add_argument('image');q.add_argument('checkpoint')
    args=p.parse_args()
    if args.command=='features':print(build_features(args.manifest,args.out).shape)
    elif args.command=='train':print(json.dumps(train(args.features,args.out),indent=2))
    else:print(predict(args.image,args.checkpoint))
