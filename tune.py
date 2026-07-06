"""Quick tuning harness: run the 4 methods via the shared-seed driver, print Dice + time."""
import os, sys, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
import SimpleITK as sitk
from brats_pipeline.seg_metrics import dice
from brats_pipeline import seg_spline_levelset as M

IMG = "images"
cases = sys.argv[1:] or sorted(os.listdir(IMG))[:5]

def load(cid, mod):
    h = glob.glob(os.path.join(IMG, cid, f"*-{mod}.nii*"))
    return sitk.GetArrayFromImage(sitk.ReadImage(h[0], sitk.sitkFloat32))

print(f"{'case':22s} {'seed':>6s} {'level':>6s} {'varspl':>6s} {'bspl':>6s} {'spline':>6s} {'t(s)':>5s}", flush=True)
agg = {k: [] for k in ["seed","level_set","variational_spline","bspline","spline"]}
for cid in cases:
    t1c = load(cid, "t1c"); t1n = load(cid, "t1n")
    gt = (np.round(load(cid, "seg")).astype(np.int16) == 3).astype(np.uint8)
    t0 = time.time()
    roi, mapa = M.roi_et_auto(t1c, t1c, t1n)
    res = M.correr_spline_levelset(t1c, t1c, t1n, verbose=False)
    dt = time.time() - t0
    d_seed = dice(roi, gt)
    ds = {k: dice(v, gt) for k, v in res.items()}
    agg["seed"].append(d_seed)
    for k in res: agg[k].append(ds[k])
    print(f"{cid:22s} {d_seed:6.3f} {ds['level_set']:6.3f} "
          f"{ds['variational_spline']:6.3f} {ds['bspline']:6.3f} {ds['spline']:6.3f} {dt:5.1f}", flush=True)

print("-"*70, flush=True)
print(f"{'MEAN':22s} {np.mean(agg['seed']):6.3f} {np.mean(agg['level_set']):6.3f} "
      f"{np.mean(agg['variational_spline']):6.3f} {np.mean(agg['bspline']):6.3f} "
      f"{np.mean(agg['spline']):6.3f}", flush=True)
print("DONE", flush=True)
