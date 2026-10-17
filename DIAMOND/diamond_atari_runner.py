"""Run from the official DIAMOND repository, exclusively inside a Slurm GPU job."""
import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from atari_protocol import TASKS, HORIZON, conditions, metrics, summarize

def sha(data):
    return hashlib.sha256(data).hexdigest()

EXPERIMENT_METADATA = {}

def policy_sensor(obs, condition, level, rng):
    """Policy-only sensor channel; the latent WM history remains unchanged."""
    if condition != 'noise':
        return obs
    import torch
    noise = rng.normal(0, level, tuple(obs.shape))
    return (obs + torch.tensor(noise, device=obs.device, dtype=obs.dtype)*2).clamp(-1,1)

def source_provenance(root):
    """Record source hashes even when shared-storage ownership blocks git metadata."""
    root = Path(root)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = sorted(list((root/'src').rglob('*.py')) + list((root/'config').rglob('*.yaml')))
    hashes = {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in files}
    return dict(git_commit=git.stdout.strip() if git.returncode == 0 else None,
                git_metadata_warning=git.stderr.strip() if git.returncode else None,
                source_sha256=hashes)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=['smoke', 'full'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU computation must run inside Slurm')
    sys.path.insert(0, str(Path('src').resolve()))
    import numpy as np
    import torch
    import gymnasium as gym
    from huggingface_hub import hf_hub_download
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import Agent
    from envs.atari_preprocessing import AtariPreprocessing
    from models.diffusion import DiffusionSampler
    from PIL import Image

    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected an allocated RTX 5090')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    args.out.mkdir(parents=True, exist_ok=True)
    revision = '5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f'
    def download(name):
        return hf_hub_download('eloialonso/diamond', name, revision=revision)
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('config').resolve())):
        cfg = compose(config_name='trainer')
    cfg.agent = OmegaConf.load(download('atari_100k/config/agent/default.yaml'))
    games = ['Breakout'] if args.profile == 'smoke' else list(TASKS)
    cases = 2 if args.profile == 'smoke' else 20
    manifest = dict(profile=args.profile, cases=cases, horizon=HORIZON, frame_skip=4,
        games=games, conditions=conditions(args.profile=='smoke'), tasks=TASKS,
        checkpoint_revision=revision, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), slurm_job=os.environ['SLURM_JOB_ID'],
        **source_provenance(Path.cwd()),
        checkpoint_sha256={}, conditioning_window=int(cfg.agent.denoiser.inner_model.num_steps_conditioning),
        snapshot_persistence={}, experiment=EXPERIMENT_METADATA,
        warning='Free-policy WM/ALE gap is diagnostic, not a calibrated early warning')
    def save_manifest():
        (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    def tensor(obs):
        return torch.as_tensor(np.array(obs), device='cuda').permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0)
    def clone_hx(hx):
        return None if hx is None else tuple(x.clone() for x in hx)
    rows = []
    for game in games:
        env = AtariPreprocessing(gym.make(f'{game}NoFrameskip-v4', full_action_space=False,
            frameskip=1, repeat_action_probability=0.0, render_mode='rgb_array', max_episode_steps=None),
            noop_max=30, frame_skip=4, screen_size=64)
        ckpt = Path(download(f'atari_100k/models/{game}.pt'))
        manifest['checkpoint_sha256'][game] = sha(ckpt.read_bytes())
        agent = Agent(instantiate(cfg.agent, num_actions=env.action_space.n)).cuda().eval()
        agent.load(ckpt)
        sampler = DiffusionSampler(agent.denoiser, instantiate(cfg.world_model_env.diffusion_sampler))
        n = manifest['conditioning_window']
        for case in range(cases):
            seed = 20261011 + case
            torch.manual_seed(seed)
            warm_rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=seed)
            hx = None
            history, actions = [tensor(obs)], []
            warm_steps = 32 + int(warm_rng.integers(0, 65))
            for t in range(warm_steps):
                prediction = agent.actor_critic.predict_act_value(history[-1], hx)
                hx = prediction.hx_cx
                probs = prediction.logits_act.softmax(-1)[0].cpu().numpy().astype(float)
                a = int(warm_rng.choice(len(probs), p=probs/probs.sum()))
                meanings = env.unwrapped.get_action_meanings()
                if t == 0 and 'FIRE' in meanings:
                    a = meanings.index('FIRE')
                obs, _, end, trunc, _ = env.step(a)
                if end or trunc:
                    raise RuntimeError(f'{game} case {case}: warmup ended; redefine pilot case bank before full run')
                actions.append(a)
                history.append(tensor(obs))
            context = torch.stack(history[-n:], dim=1)
            past_actions = torch.tensor([actions[-(n-1):]+[0]], device='cuda', dtype=torch.long)
            initial_hx = clone_hx(hx)
            # Capture emulator state and RNG using the unified ALE API.
            state = env.ale.cloneState(include_rng=True)
            native_state_bytes = pickle.dumps(state)
            # Include every mutable preprocessing field and wrapper episode counter.
            wrappers, current = [], env
            while isinstance(current, gym.Wrapper):
                fields = {k: copy.deepcopy(v) for k,v in vars(current).items()
                          if k in ('obs_buffer','lives','game_over','_elapsed_steps','_has_reset')}
                wrappers.append((current, fields))
                current = current.env
            rng_state = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)
            initial_ram = env.ale.getRAM().copy()
            fingerprint = sha(native_state_bytes+context.cpu().numpy().tobytes()+past_actions.cpu().numpy().tobytes())
            np.savez_compressed(args.out/f'{game}-case-{case}.npz', ram=initial_ram,
                context=context.cpu().numpy(), actions=past_actions.cpu().numpy(), seed=seed)
            try:
                (args.out/f'{game}-case-{case}.state.pkl').write_bytes(pickle.dumps((state, rng_state, [f for _,f in wrappers])))
                manifest['snapshot_persistence'][f'{game}/{case}'] = True
            except Exception as error:
                manifest['snapshot_persistence'][f'{game}/{case}'] = False
                print(f'Snapshot serialization unavailable; in-memory paired restore remains active: {error}', flush=True)
            normal_hashes = {}
            for condition, level in conditions(args.profile=='smoke'):
                for domain in ('ale','wm'):
                    torch.manual_seed(seed + 100000)
                    policy_rng = np.random.default_rng(seed+200000)
                    drop_rng = np.random.default_rng(seed+300000)
                    noise_rng = np.random.default_rng(seed+400000)
                    env.ale.restoreState(state)
                    env.unwrapped.np_random.bit_generator.state = copy.deepcopy(rng_state)
                    for wrapper, fields in wrappers:
                        for k,v in fields.items(): setattr(wrapper,k,copy.deepcopy(v))
                    # ALE 0.9 restoreState restores the emulator, but getRAM()
                    # exposes a stale cache until the next act(). Verify actual
                    # serialized emulator state (including RNG), not that cache.
                    if pickle.dumps(env.ale.cloneState(include_rng=True)) != native_state_bytes:
                        raise RuntimeError('ALE initial-state restore failed')
                    buf, act = context.clone(), past_actions.clone()
                    hx = clone_hx(initial_hx)
                    _, _, rhx = agent.rew_end_model.predict_rew_end(buf[:,:-1], act[:,:-1], buf[:,1:])
                    obs_tensor = buf[:,-1]
                    rewards, intended, executed, frames = [], [], [], []
                    digest = hashlib.sha256()
                    ended = False
                    for step in range(HORIZON):
                        sensor = policy_sensor(obs_tensor, condition, level, noise_rng)
                        prediction = agent.actor_critic.predict_act_value(sensor, hx)
                        hx = prediction.hx_cx
                        probs = prediction.logits_act.softmax(-1)[0].cpu().numpy().astype(float)
                        a = int(policy_rng.choice(len(probs), p=probs/probs.sum()))
                        b = 0 if condition == 'drop' and drop_rng.random() < level else a
                        if domain == 'ale':
                            obs, reward, end, trunc, _ = env.step(b)
                            obs_tensor = tensor(obs)
                            ended = bool(end or trunc)
                        else:
                            act[:,-1] = b
                            next_obs, _ = sampler.sample(buf, act)
                            lr, le, rhx = agent.rew_end_model.predict_rew_end(buf[:,-1:], act[:,-1:], next_obs.unsqueeze(1), rhx)
                            reward = float(torch.distributions.Categorical(logits=lr).sample().item()-1)
                            ended = bool(torch.distributions.Categorical(logits=le).sample().item())
                            buf = torch.cat([buf[:,1:], next_obs.unsqueeze(1)], dim=1)
                            act = torch.cat([act[:,1:], torch.zeros_like(act[:,:1])], dim=1)
                            obs_tensor = next_obs
                        if not torch.isfinite(obs_tensor).all():
                            raise RuntimeError('Non-finite observation')
                        pixels = ((obs_tensor[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()
                        digest.update(pixels.tobytes())
                        digest.update(json.dumps([a,b,float(reward),ended]).encode())
                        rewards.append(float(reward)); intended.append(a); executed.append(b)
                        if case == 0 and condition in ('normal','sham'):
                            frames.append(Image.fromarray(pixels).resize((256,256), Image.Resampling.NEAREST))
                        if ended: break  # no auto-reset and no second episode within a task
                    trajectory_hash = digest.hexdigest()
                    if condition == 'normal': normal_hashes[domain] = trajectory_hash
                    if condition == 'sham' and trajectory_hash != normal_hashes[domain]:
                        raise RuntimeError('SHAM mismatch: paired repeat is not reproducible')
                    row = dict(game=game, case=case, seed=seed, domain=domain, condition=condition,
                        level=level, initial_state_sha256=fingerprint, trajectory_sha256=trajectory_hash,
                        ended=ended, **metrics(rewards, TASKS[game]['positive_steps_required']))
                    if domain == 'wm': row['raw_return'] = None  # reward head only predicts clipped rewards
                    rows.append(row)
                    with (args.out/'trials.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(row, rewards=rewards, intended_actions=intended, executed_actions=executed))+'\n')
                    if frames:
                        frames[0].save(args.out/f'{game}-{domain}-{condition}.gif', save_all=True,
                            append_images=frames[1:], duration=67, loop=0)
                    print(json.dumps(row), flush=True)
            save_manifest()
        env.close()
        del agent, sampler
        torch.cuda.empty_cache()
    summary = summarize(rows, cases)
    with (args.out/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    manifest['status'] = 'passed'
    manifest['trials_completed'] = len(rows)
    save_manifest()
    (args.out/'PASS').write_text('All cases complete; normal/sham trajectory hashes match.\n')
    print('DIAMOND_ATARI_TASKS_PASSED', flush=True)

if __name__ == '__main__':
    # Importing torch happens after Slurm validation in main.
    import traceback
    try:
        # Inference mode also prevents recurrent state graphs from growing during long rollouts.
        import torch
        with torch.inference_mode(): main()
    except Exception:
        traceback.print_exc()
        raise
