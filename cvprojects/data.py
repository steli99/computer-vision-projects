"""Dataset downloads and safe, CRC-checked ZIP extraction."""
from pathlib import Path
import struct,zlib,zipfile,urllib.request

COURSE_BASE='https://lttm.dei.unipd.it/downloads/prj2025/'

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    urllib.request.urlretrieve(url,temp)
    temp.replace(path)
    return path

def safe_target(root,name):
    root=Path(root).resolve();target=(root/name).resolve()
    if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
    return target

def extract_zip(path,destination,predicate=lambda name:True):
    """Recover complete local members if the central directory is missing.

    Recovery requires explicit sizes (no data descriptor) and checks each selected
    member's decompressed size and CRC. Truncated members are never written.
    """
    paths=[];recovered=False
    try:
        archive=zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        archive=None;recovered=True
    if archive is not None:
        with archive:
            for info in archive.infolist():
                if info.is_dir() or not predicate(info.filename):continue
                target=safe_target(destination,info.filename);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(info));paths.append(target)
    else:
        with open(path,'rb') as f:
            while True:
                header=f.read(30)
                if len(header)!=30 or header[:4]!=b'PK\x03\x04':break
                _,version,flags,method,tm,date,crc,csize,usize,nlen,xlen=struct.unpack('<IHHHHHIIIHH',header)
                name=f.read(nlen).decode('utf-8' if flags & 2048 else 'cp437');f.read(xlen)
                if flags & 9:raise ValueError('Cannot recover encrypted/data-descriptor ZIP members')
                payload=f.read(csize)
                if len(payload)!=csize:break
                if name.endswith('/') or not predicate(name):continue
                if method==0:data=payload
                elif method==8:data=zlib.decompress(payload,-15)
                else:raise ValueError('Unsupported recovery compression')
                if len(data)!=usize or zlib.crc32(data)&0xffffffff!=crc:raise ValueError(f'CRC failure: {name}')
                target=safe_target(destination,name);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data);paths.append(target)
    return paths,recovered

def prepare_isic(destination,per_class=500,seed=42,workers=8):
    """Reproducible balanced ISIC-2019 MEL/NV subset from the specified Kaggle dataset.

    All included labels and lesion identifiers are the official ISIC 2019 CSVs.
    No authentication is synthesized; if public access fails, the error is reported.
    """
    import pandas as pd
    import concurrent.futures,urllib.parse,time
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    for file in ['ISIC_2019_Training_GroundTruth.csv','ISIC_2019_Training_Metadata.csv']:
        if not (root/file).exists():download('https://isic-challenge-data.s3.amazonaws.com/2019/'+file,root/file)
    labels=pd.read_csv(root/'ISIC_2019_Training_GroundTruth.csv')
    meta=pd.read_csv(root/'ISIC_2019_Training_Metadata.csv')
    binary=labels[(labels.MEL==1)|(labels.NV==1)].copy();binary['label']=binary.MEL.astype(int)
    if per_class:
        binary=binary.groupby('label',group_keys=False).sample(n=per_class,random_state=seed)
    binary=binary[['image','label']].merge(meta[['image','lesion_id']],on='image',how='left')
    binary['group']=binary.lesion_id.fillna(binary.image)
    binary['path']='images/'+binary.image+'.jpg'
    (root/'images').mkdir(exist_ok=True)
    def fetch(row):
        target=root/row.path
        if target.exists():return row.image
        name='ISIC_2019_Training_Input/ISIC_2019_Training_Input/'+row.image+'.jpg'
        url='https://api.kaggle.com/v1/datasets/download/andrewmvd/isic-2019/'+urllib.parse.quote(name,safe='')
        last=None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
                if not data.startswith(b'\xff\xd8'):raise ValueError('Response is not a JPEG')
                target.write_bytes(data);return row.image
            except Exception as exc:
                last=exc
                if attempt<2:time.sleep(1+attempt)
        raise RuntimeError(f'Image download failed: {row.image}: {last}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(fetch,binary.itertuples(index=False)),1):
            if i%100==0:print('ISIC images downloaded:',i,flush=True)
    binary.to_csv(root/'manifest.csv',index=False)
    return root/'manifest.csv'
