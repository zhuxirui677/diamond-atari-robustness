"""DIAMOND with upstream LIBERO-plus Sensor Noise, through the paired runner."""
import hashlib
import json
import os
from pathlib import Path

FAMILIES = ('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')

def plus_conditions(smoke=False, family=None):
    family = family or os.environ.get('DIAMOND_PLUS_FAMILY','gaussian_blur')
    if family not in FAMILIES:
        raise ValueError('Unknown Sensor Noise family: '+family)
    levels = (1,10) if smoke else range(1,11)
    return [('normal',0.),('sham',0.)] + [('plus_'+family,float(i)) for i in levels]

def transform_rgb(image, condition, severity, rng=None):
    """No uint8 round trip for normal/sham; caller bypasses those conditions."""
    import numpy as np
    from PIL import Image
    import libero_plus_image_functions as functions
    family = condition.removeprefix('plus_')
    if family not in FAMILIES or not condition.startswith('plus_'):
        raise ValueError('Unsupported plus condition: '+condition)
    if int(severity) != severity or not 1 <= severity <= 10:
        raise ValueError('LIBERO-plus severity must be 1..10')
    if family=='motion_blur' and not hasattr(functions,'MotionImage'):
        functions.enable_motion_blur()
    # Upstream stochastic functions use the legacy global NumPy RNG.
    # Isolate that state so perturbations do not alter policy or emulator RNGs.
    previous=np.random.get_state()
    seed=int(rng.integers(0,2**32)) if rng is not None else 0
    try:
        np.random.seed(seed)
        result=getattr(functions,family)(Image.fromarray(image),int(severity))
        if result.shape != image.shape or not np.isfinite(result).all():
            raise RuntimeError('Invalid upstream sensor output')
        return result.astype(np.uint8)
    finally:
        np.random.set_state(previous)

def plus_sensor(obs, condition, level, rng):
    if condition in ('normal','sham'):
        return obs
    import torch
    pixels = ((obs[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()
    changed = transform_rgb(pixels,condition,level,rng)
    return torch.as_tensor(changed,device=obs.device).permute(2,0,1).to(obs.dtype).div(127.5).sub(1).unsqueeze(0)

def main():
    import diamond_atari_runner as runner
    root=Path(__file__).resolve().parent
    provenance=json.loads((root/'libero_plus_provenance.json').read_text())
    family=os.environ.get('DIAMOND_PLUS_FAMILY','gaussian_blur')
    # Fail visibly before rollout when a requested native backend is unavailable.
    import numpy as np
    transform_rgb(np.full((64,64,3),127,dtype=np.uint8),'plus_'+family,1)
    provenance.update(sensor_family=family,severities=list(range(1,11)),
                      stochastic_rng='isolated NumPy state; per-case perturbation RNG')
    provenance['local_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('diamond_libero_plus_runner.py','libero_plus_image_functions.py','diamond_atari_runner.py','atari_protocol.py')}
    runner.conditions=plus_conditions
    runner.policy_sensor=plus_sensor
    runner.EXPERIMENT_METADATA=provenance
    runner.main()

if __name__=='__main__':
    import torch
    with torch.inference_mode():
        main()
