"""Run from repository root: python scripts/download_data.py roads sequences weights isic."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from cvprojects.data import download,extract_zip,prepare_isic,COURSE_BASE
p=argparse.ArgumentParser();p.add_argument('datasets',nargs='+',choices=['roads','sequences','weights','isic','video']);p.add_argument('--root',default='data');p.add_argument('--per-class',type=int,default=150);p.add_argument('--workers',type=int,default=8)
a=p.parse_args();root=Path(a.root)
for dataset in a.datasets:
    if dataset=='isic':print(prepare_isic(root/'isic',a.per_class,workers=a.workers))
    elif dataset=='weights':
        for name in ['deploy.prototxt','mobilenet_iter_73000.caffemodel']:
            target=root/'weights'/name
            if not target.exists():download('https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/'+name,target)
    elif dataset=='video':download(COURSE_BASE+'video1.mp4',root/'video1.mp4')
    else:
        names=['roads'] if dataset=='roads' else ['easy','challenging']
        for name in names:
            archive=root/'downloads'/f'{name}.zip'
            if not archive.exists():download(COURSE_BASE+name+'.zip',archive)
            target=root/'roads' if dataset=='roads' else root/'sequences'/name
            predicate=(lambda n:n.lower().endswith(('.jpg','.png','.jpeg'))) if dataset=='roads' else (lambda n:'/upper_loop/rgb/' in '/'+n and n.lower().endswith('.png'))
            files,recovered=extract_zip(archive,target,predicate)
            # Course archives may contain a wrapping scene directory: flatten to documented location.
            desired=target if dataset=='roads' else target/'upper_loop'/'rgb'
            desired.mkdir(parents=True,exist_ok=True)
            for file in files:
                if file.parent!=desired:file.replace(desired/file.name)
            if dataset=='sequences' and len(files)!=25:raise RuntimeError(f'Expected 25 RGB frames, got {len(files)}')
            print(name,len(files),'CRC-checked local-header recovery' if recovered else 'normal extraction')
