import unittest,tempfile,zipfile
from pathlib import Path
import numpy as np
import cv2
from cvprojects.sequence import solve_cycle,cycle_metrics
from cvprojects.data import extract_zip,safe_target
from cvprojects.lanes import CrossingTracker,assign_lane,marking_type
from cvprojects.lesions import shape_features,masked_texture,segment

class Algorithms(unittest.TestCase):
    def test_cycle_recovers_strong_ring(self):
        n=9;s=np.ones((n,n))*.01;np.fill_diagonal(s,0)
        for i in range(n):s[i,(i+1)%n]=s[(i+1)%n,i]=10
        order,diagnostics=solve_cycle(s)
        self.assertEqual(cycle_metrics(order,list(range(n)))['undirected_cycle_edge_recall'],1)
        self.assertEqual(diagnostics['unsupported_cycle_edges'],0)
    def test_cycle_rotation_and_reversal(self):
        self.assertEqual(cycle_metrics([2,1,0,4,3],list(range(5)))['position_accuracy_up_to_rotation_and_reversal'],1)
    def test_disconnected_cycle_flagged(self):
        _,d=solve_cycle(np.zeros((4,4)));self.assertEqual(d['confidence'],'low')
    def test_archive_recovery_and_crc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.zip'
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('upper_loop/rgb/a.txt',b'valid pixels')
            data=path.read_bytes();path.write_bytes(data[:data.index(b'PK\x01\x02')])
            files,recovered=extract_zip(path,Path(tmp)/'out')
            self.assertTrue(recovered);self.assertEqual(files[0].read_bytes(),b'valid pixels')
            corrupt=bytearray(path.read_bytes());corrupt[14]^=1;path.write_bytes(corrupt)
            with self.assertRaises(ValueError):extract_zip(path,Path(tmp)/'bad')
    def test_archive_traversal_rejected(self):
        with self.assertRaises(ValueError):safe_target('/tmp/test-root','../escape')
    def test_lane_assignment(self):
        lanes=[dict(id=1,left=dict(a=-.5,b=50),right=dict(a=.5,b=50))]
        self.assertEqual(assign_lane(50,80,lanes),1);self.assertIsNone(assign_lane(100,80,lanes))
    def test_gradual_crossing_and_missing_frame_reset(self):
        tracker=CrossingTracker();events=[]
        for x in [47,49.5,50,50.5,53,54,54,54,54]:events.append(tracker.update([dict(a=0,b=x)],(100,100)))
        self.assertFalse(any(events[:5]));self.assertEqual(sum(events),1)
        tracker.update([],(100,100));self.assertFalse(tracker.update([dict(a=0,b=47)],(100,100)))
    def test_marking_types(self):
        solid=np.zeros((200,100),np.uint8);solid[:,48:53]=255
        dashed=solid.copy()
        for y in [40,90,140]:dashed[y:y+20]=0
        self.assertEqual(marking_type(solid,0,50,0,199)[0],'solid')
        self.assertEqual(marking_type(dashed,0,50,0,199)[0],'dashed')
    def test_circle_symmetry(self):
        m=np.zeros((128,128),np.uint8);cv2.circle(m,(64,64),30,1,-1)
        features=shape_features(m.astype(bool))
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertLess(features['asymmetry_major'],.08)
    def test_segmentation_with_dark_border_and_small_lesion(self):
        image=np.zeros((200,200,3),np.uint8)
        cv2.circle(image,(100,100),88,(220,170,145),-1)
        cv2.circle(image,(100,100),12,(85,45,30),-1)
        mask=segment(image)
        self.assertTrue(mask[100,100]);self.assertFalse(mask[0,0])
        self.assertLess(mask.mean(),.15)
    def test_texture_ignores_background(self):
        mask=np.zeros((60,60),bool);mask[15:45,15:45]=True
        a=np.full((60,60),120,np.uint8);b=a.copy();b[~mask]=0
        self.assertEqual(masked_texture(a,mask),masked_texture(b,mask))

if __name__=='__main__':unittest.main()
