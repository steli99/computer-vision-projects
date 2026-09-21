"""Project 3: geometrically verified image matching and closed-loop ordering."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components


def load_gray(path,max_side=720):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        image=np.load(path)
        if image.ndim==3:image=np.mean(image,axis=-1)
        low,high=np.percentile(image,[1,99]);image=np.uint8(np.clip((image-low)/(high-low+1e-8)*255,0,255))
    else:image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError(f'Cannot read {path}')
    scale=min(1,max_side/max(image.shape))
    if scale<1:image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    return image


def extract(path,method='sift',max_side=720):
    gray=load_gray(path,max_side)
    detector=cv2.SIFT_create(nfeatures=1800) if method=='sift' else cv2.ORB_create(nfeatures=2500)
    keypoints,descriptors=detector.detectAndCompute(gray,None)
    return {'points':np.array([k.pt for k in keypoints],dtype=np.float32).reshape(-1,2),
            'descriptors':descriptors,'shape':gray.shape}


def reciprocal_ratio_matches(a,b,method='sift',ratio=.75):
    if a is None or b is None or len(a)<2 or len(b)<2:return []
    matcher=cv2.BFMatcher(cv2.NORM_L2 if method=='sift' else cv2.NORM_HAMMING)
    forward=matcher.knnMatch(a,b,k=2);backward=matcher.knnMatch(b,a,k=2)
    good_backward={(m.queryIdx,m.trainIdx) for pair in backward if len(pair)==2 for m,n in [pair] if m.distance<ratio*n.distance}
    return [m for pair in forward if len(pair)==2 for m,n in [pair]
            if m.distance<ratio*n.distance and (m.trainIdx,m.queryIdx) in good_backward]


def pair_score(a,b,method='sift'):
    matches=reciprocal_ratio_matches(a['descriptors'],b['descriptors'],method)
    result={'matches':len(matches),'inliers':0,'ratio':0.,'coverage':0.,'score':0.}
    if len(matches)<12:return result
    p=np.float32([a['points'][m.queryIdx] for m in matches]);q=np.float32([b['points'][m.trainIdx] for m in matches])
    cv2.setRNGSeed(42)
    _,mask=cv2.findFundamentalMat(p,q,cv2.FM_RANSAC,1.5,.995)
    if mask is None or mask.size!=len(matches):return result
    good=mask.ravel().astype(bool);count=int(good.sum())
    if count<12:return result
    coverage=[]
    for points,shape in [(p[good],a['shape']),(q[good],b['shape'])]:
        height,width=shape
        bins=np.clip((points/np.array([width,height])*4).astype(int),0,3)
        coverage.append(len(np.unique(bins,axis=0))/16)
    cov=float(np.sqrt(np.prod(coverage)))
    ratio=count/len(matches)
    # Good adjacent views usually share many geometrically consistent features.
    score=float(np.sqrt(count)*ratio*np.sqrt(cov))
    result.update(inliers=count,ratio=ratio,coverage=cov,score=score)
    return result


def cycle_cost(order,cost):return float(sum(cost[order[i],order[(i+1)%len(order)]] for i in range(len(order))))


def two_opt(order,cost,max_passes=100):
    order=list(order);n=len(order)
    for _ in range(max_passes):
        improved=False
        for i in range(n-2):
            for j in range(i+2,n):
                if i==0 and j==n-1:continue
                a,b,c,d=order[i],order[i+1],order[j],order[(j+1)%n]
                if cost[a,c]+cost[b,d] < cost[a,b]+cost[c,d]-1e-10:
                    order[i+1:j+1]=reversed(order[i+1:j+1]);improved=True
        if not improved:break
    return order


def solve_cycle(scores):
    """Multi-start Hamiltonian-cycle heuristic; rotation/direction are unidentifiable."""
    scores=np.asarray(scores,float);n=len(scores)
    if scores.shape!=(n,n) or n<3 or not np.allclose(scores,scores.T):raise ValueError('Need a symmetric NxN matrix, N>=3')
    if not np.isfinite(scores).all() or np.any(scores<0):raise ValueError('Scores must be finite and nonnegative')
    max_score=max(float(scores.max()),1e-8)
    cost=-np.log(np.maximum(scores/max_score,1e-6));np.fill_diagonal(cost,0)
    candidates=[]
    for start in range(n):
        path=[start];unvisited=set(range(n))-{start}
        while unvisited:
            nxt=min(unvisited,key=lambda k:(cost[path[-1],k],k));path.append(nxt);unvisited.remove(nxt)
        candidates.append(two_opt(path,cost))
        # Cheapest insertion is a second initialization, avoiding greedy dead ends.
        nxt=min((k for k in range(n) if k!=start),key=lambda k:cost[start,k])
        path=[start,nxt];remaining=set(range(n))-set(path)
        while remaining:
            _,item,pos=min((cost[path[i],k]+cost[k,path[(i+1)%len(path)]]-cost[path[i],path[(i+1)%len(path)]],k,i+1)
                           for k in remaining for i in range(len(path)))
            path.insert(pos,item);remaining.remove(item)
        candidates.append(two_opt(path,cost))
    order=min(candidates,key=lambda o:cycle_cost(o,cost))
    components,_=connected_components(scores>0,directed=False)
    unsupported=sum(scores[order[i],order[(i+1)%n]]<=0 for i in range(n))
    return order,{'cost':cycle_cost(order,cost),'connected_components':int(components),
                  'unsupported_cycle_edges':int(unsupported),'confidence':'low' if unsupported or components>1 else 'geometrically_supported',
                  'solver':'multi-start nearest-neighbor + insertion + 2-opt (heuristic, not exact)'}


def cycle_metrics(predicted,truth):
    if set(predicted)!=set(truth) or len(set(truth))!=len(truth):raise ValueError('Orders must contain the same unique images')
    def edges(order):return {frozenset((order[i],order[(i+1)%len(order)])) for i in range(len(order))}
    edge_recall=len(edges(predicted)&edges(truth))/len(truth)
    best=max(sum(a==b for a,b in zip(predicted,list(direction)[k:]+list(direction)[:k]))/len(truth)
             for direction in [truth,list(reversed(truth))] for k in range(len(truth)))
    return {'undirected_cycle_edge_recall':edge_recall,'position_accuracy_up_to_rotation_and_reversal':best}


def run(image_dir,outdir,method='sift',seed=42,truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    files=[p for p in Path(image_dir).iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.npy']]
    files=sorted(files);np.random.default_rng(seed).shuffle(files) # never use capture-name adjacency for inference
    if len(files)<3:raise ValueError('At least three images are required')
    features=[extract(p,method) for p in files];n=len(files);scores=np.zeros((n,n));pairs=[]
    for i in range(n):
        for j in range(i+1,n):
            record=pair_score(features[i],features[j],method);scores[i,j]=scores[j,i]=record['score']
            pairs.append(dict(a=files[i].name,b=files[j].name,**record))
    order,diagnostics=solve_cycle(scores);names=[files[i].name for i in order]
    result={'order':names,'input_order':[p.name for p in files],'input_shuffle_seed':seed,'method':method,**diagnostics}
    if truth is not None:result['evaluation']=cycle_metrics(names,list(truth))
    (out/'order.json').write_text(json.dumps(result,indent=2));(out/'pair_matches.json').write_text(json.dumps(pairs,indent=2))
    np.save(out/'similarity.npy',scores)
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(scores[np.ix_(order,order)],cmap='magma');fig.colorbar(im,ax=ax)
    ax.set_title('Geometric similarity, reordered by estimated cycle');ax.set_xlabel('Estimated position');ax.set_ylabel('Estimated position')
    fig.tight_layout();fig.savefig(out/'similarity.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(int(np.ceil(n/5)),5,figsize=(13,2.5*np.ceil(n/5)),squeeze=False)
    for k,ax in enumerate(axes.ravel()):
        ax.axis('off')
        if k<n:
            p=files[order[k]];image=cv2.imread(str(p))
            if image is not None:ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            else:ax.imshow(load_gray(p),cmap='gray')
            ax.set_title(f'{k+1}: {p.name}',fontsize=9)
    fig.tight_layout();fig.savefig(out/'estimated_sequence.jpg',dpi=110);plt.close(fig)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('images');parser.add_argument('--out',default='results/project3');parser.add_argument('--method',choices=['sift','orb'],default='sift');parser.add_argument('--truth',help='JSON ordered filename list, used for evaluation only')
    args=parser.parse_args();truth=json.loads(Path(args.truth).read_text()) if args.truth else None
    print(json.dumps(run(args.images,args.out,args.method,truth=truth),indent=2))
