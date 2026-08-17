"""How much gaze noise can corner-look calibration tolerate? The model's own
reported error is ~2.4 deg, so that column is the realistic one."""
# ruff: noqa: E402  -- sys.path must be set before the package import
import sys

import numpy as np

sys.path.insert(0,"/home/claude/argus_gaze_port")
from gaze.calibrate import CornerCollector
from gaze.geometry import CalibrationError

w,h = 597.0,336.0
yaw,tilt = np.deg2rad(25), np.deg2rad(-8)
Ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
Rx=np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
R=Ry@Rx
tl=np.array([-250.,-180.,420.])
truth=np.array([tl, tl+R@[w,0,0], tl+R@[w,h,0], tl+R@[0,h,0]])

def trial(noise_deg, n_pos, seed):
    rng=np.random.default_rng(seed)
    pos=[np.array([0.,0.,0.])]+[rng.normal(0,70,3)+np.array([0,0,rng.normal(0,50)]) for _ in range(n_pos-1)]
    c=CornerCollector("s",(2560,1440))
    for eye in pos:
        for i in range(4):
            d=truth[i]-eye
            d/=np.linalg.norm(d)
            d=d+rng.normal(0,np.deg2rad(noise_deg),3)
            d/=np.linalg.norm(d)
            c.add(i,eye,d)
    p,r=c.solve()
    got=np.array([p.origin,p.origin+p.x_axis*p.width_mm,
                  p.origin+p.x_axis*p.width_mm+p.y_axis*p.height_mm,
                  p.origin+p.y_axis*p.height_mm])
    return float(np.mean(np.linalg.norm(got-truth,axis=1))), r["diagonal_in"]

print(f"{'gaze noise':>11} | {'samples/corner':>14} | {'mean corner err':>16} | {'solved diag':>11} | {'ok'}")
print("-"*72)
for noise in (0.0,0.5,1.0,2.4,4.0):
    for n in (4,8,16):
        errs,diags,fail=[],[],0
        for s in range(40):
            try:
                e,d=trial(noise,n,s)
                errs.append(e)
                diags.append(d)
            except CalibrationError:
                fail+=1
        if errs:
            print(f"{noise:>9.1f}d | {n:>14} | {np.median(errs):>13.0f} mm | {np.median(diags):>9.1f}in | {40-fail}/40")
        else:
            print(f"{noise:>9.1f}d | {n:>14} | {'--':>16} | {'--':>11} | 0/40 (all refused)")
