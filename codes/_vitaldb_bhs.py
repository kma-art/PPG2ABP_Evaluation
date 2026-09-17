"""Compute BHS and AAMI for VitalDB Variant A (all 9400 segments)."""
import numpy as np, pickle, os, torch
from _models_pytorch import UNetDS64, MultiResUNet1D

raw = pickle.load(open(os.path.join('raw_data', 'vitaldb_raw.p'), 'rb'))
X_raw, Y_raw = raw['X_raw'], raw['Y_raw']
meta = pickle.load(open(os.path.join('data', 'meta9.p'), 'rb'))
min_abp, max_abp = meta['min_abp'], meta['max_abp']

X_norm = ((X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
approx = UNetDS64(1, 1).to(device)
approx.load_state_dict(torch.load('models/approximate_state_dict.pth', map_location=device, weights_only=True))
approx.eval()
refine = MultiResUNet1D(n_channel=1).to(device)
refine.load_state_dict(torch.load('models/refinement_state_dict.pth', map_location=device, weights_only=True))
refine.eval()

X_t = torch.tensor(X_norm.reshape(-1, 1024, 1), dtype=torch.float32)
preds = []
for s in range(0, len(X_t), 500):
    b = X_t[s:s+500].to(device)
    with torch.no_grad():
        a = approx(b)
        y = a[0].permute(0, 2, 1).contiguous()
        r = refine(y).permute(0, 2, 1).contiguous()
    preds.append(r.cpu().numpy() * (max_abp - min_abp) + min_abp)

pred = np.concatenate([p[:, :, 0] for p in preds])

sbp_err = np.abs(pred.max(1) - Y_raw.max(1))
dbp_err = np.abs(pred.min(1) - Y_raw.min(1))
map_err = np.abs(pred.mean(1) - Y_raw.mean(1))

print("BHS (all 9400 segments, Variant A):")
for name, err in [('SBP', sbp_err), ('DBP', dbp_err), ('MAP', map_err)]:
    print(f'  {name}: <=5={((err<=5).mean()*100):.1f}%  <=10={((err<=10).mean()*100):.1f}%  <=15={((err<=15).mean()*100):.1f}%')

sbp_s = pred.max(1) - Y_raw.max(1)
dbp_s = pred.min(1) - Y_raw.min(1)
map_s = pred.mean(1) - Y_raw.mean(1)
print("\nAAMI (all 9400 segments, Variant A):")
print(f'  SBP: {sbp_s.mean():.2f} +/- {sbp_s.std():.2f}')
print(f'  DBP: {dbp_s.mean():.2f} +/- {dbp_s.std():.2f}')
print(f'  MAP: {map_s.mean():.2f} +/- {map_s.std():.2f}')
