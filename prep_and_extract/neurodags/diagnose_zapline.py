"""Is ZapLine actually removing the line on LSD? Compare PickedRaw vs DenoisedRaw,
and re-run ZapLine live to see n_removed_. Also check a ketamine file for contrast."""
import glob, re
import numpy as np
import mne
from scipy.signal import welch
from mne_denoise.zapline import ZapLine

mne.set_log_level("error")
DD = "/scratch/yorguin/cocodelics_derivatives"


def ratio50(raw, f0=50.0):
    picks = [c for c in raw.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
    d = raw.copy().pick(picks).get_data()
    sf = float(raw.info["sfreq"])
    f, p = welch(d, fs=sf, nperseg=int(sf * 2), axis=-1)
    p = p.mean(0)
    peak = p[(f >= f0 - 1) & (f <= f0 + 1)].max()
    base = np.median(p[((f >= f0 - 5) & (f <= f0 - 2)) | ((f >= f0 + 2) & (f <= f0 + 5))])
    return peak / base, sf


for ds, pat in [("MEG_LSD", "*sub-S3LR*task-Open2*"), ("MEG_ketamine", "*")]:
    pk = sorted(glob.glob(f"{DD}/{ds}/**/{pat}@PickedRaw.fif", recursive=True))
    dn = sorted(glob.glob(f"{DD}/{ds}/**/{pat}@DenoisedRaw.fif", recursive=True))
    if not pk or not dn:
        print(f"{ds}: missing ({len(pk)} picked, {len(dn)} denoised)"); continue
    rawp = mne.io.read_raw_fif(pk[0], preload=True, verbose="error")
    rawd = mne.io.read_raw_fif(dn[0], preload=True, verbose="error")
    rp, sf = ratio50(rawp)
    rd, _ = ratio50(rawd)
    print(f"\n### {ds}  {pk[0].split('/')[-1][:40]}  sfreq={sf:g}")
    print(f"  50Hz ratio: PickedRaw={rp:.2f}  DenoisedRaw={rd:.2f}  (reduction {100*(1-rd/rp):.0f}%)")
    # re-run ZapLine live with the node config to see n_removed_
    picks = [c for c in rawp.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
    data = rawp.copy().pick(picks).get_data()
    scale = float(np.std(data)) or 1.0
    est = ZapLine(sfreq=sf, line_freq=50.0, n_harmonics=2, adaptive=True)
    cleaned = est.fit_transform(data / scale) * scale
    nrem = est.n_removed_
    if getattr(est, "adaptive_results_", None):
        nrem = est.adaptive_results_.get("n_removed", nrem)
    f, po = welch(data, fs=sf, nperseg=int(sf * 2), axis=-1); po = po.mean(0)
    _, pc = welch(cleaned, fs=sf, nperseg=int(sf * 2), axis=-1); pc = pc.mean(0)
    def r(p):
        pk_ = p[(f >= 49) & (f <= 51)].max(); bs = np.median(p[((f>=45)&(f<=48))|((f>=52)&(f<=55))]); return pk_/bs
    print(f"  live ZapLine re-run: n_removed={nrem}  50Hz {r(po):.2f} -> {r(pc):.2f}")
