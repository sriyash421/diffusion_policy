"""Synthetic (no simulator) tests for the SimplerEnv Veritas verifier stack."""
import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.env.simpler.veritas.camera import CameraModel, project, quat_wxyz_to_R  # noqa: E402
from diffusion_policy.env.simpler.veritas.tracker import (  # noqa: E402
    EMPTY_TARGETS, SimplerEpisodeTracker, SimplerTrackerSnapshotStore, decode_targets,
    targets_vector)
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier  # noqa: E402
from diffusion_policy.veritas.plan_store import (  # noqa: E402
    CachedPlanProvider, has_plan, link_seed, load_plan, write_plan_record)
from diffusion_policy.veritas.schemas import VeritasPlan, VeritasWaypoint  # noqa: E402
from diffusion_policy.veritas.tracker import (  # noqa: E402
    TrackerParams, WaypointTracker, anchored_from_plan)


# ------------------------------------------------------------------ fixtures

def _camera(tmp_path=None):
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
    # camera looking down -z_world from 1 m above the origin: world (x, y, 0) -> cam (x, -y, 1)
    T = np.eye(4)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    T[2, 3] = 1.0
    base = np.array([0, 0, 0, 0, 0, 0, 1.0])  # yaw 180 deg -> R = diag(-1, -1, 1)
    cam = CameraModel(K, T, (480, 640), base)
    if tmp_path is not None:
        path = tmp_path / 'cam.json'
        cam.save(path)
        return cam, path
    return cam


def _plan(uvs, tol=60.0, min_hold=1, skippable=True):
    wps = [VeritasWaypoint(uv=tuple(uv), tol_px=tol, min_hold=min_hold, skippable=skippable)
           for uv in uvs]
    return VeritasPlan(waypoints=wps)


# ------------------------------------------------------------------ camera

def test_projection_principal_point_and_behind():
    cam = _camera()
    assert np.allclose(cam.project(np.array([0.0, 0.0, 0.0])), [320.0, 240.0])
    # +x world -> +u; +y world -> -v (because of the axis flip in T)
    uv = cam.project(np.array([0.1, 0.1, 0.0]))
    assert uv[0] > 320 and uv[1] < 240
    assert np.isnan(cam.project(np.array([0.0, 0.0, 2.0]))).all()   # behind the camera


def test_quat_and_base_rotation():
    assert np.allclose(quat_wxyz_to_R([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(_camera().R_world_from_base, np.diag([-1.0, -1.0, 1.0]))


def test_camera_roundtrip_and_assert(tmp_path):
    cam, path = _camera(tmp_path)
    loaded = CameraModel.load(path)
    assert np.allclose(loaded.K, cam.K) and np.allclose(loaded.T_wc, cam.T_wc)
    obs = {'camera_param': {'3rd_view_camera': {'intrinsic_cv': cam.K, 'extrinsic_cv': cam.T_wc}},
           'image': {'3rd_view_camera': {'rgb': np.zeros((480, 640, 3), np.uint8)}},
           'agent': {'base_pose': cam.base_pose}}
    loaded.assert_matches(obs)
    obs['agent']['base_pose'] = cam.base_pose + np.array([0.01, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        loaded.assert_matches(obs)


# ------------------------------------------------------------------ tracker

def test_tracker_params_scale_matches_pusht_constants():
    p = TrackerParams.for_image_width(96)
    assert math.isclose(p.sigma_px, 7.5) and math.isclose(p.skip_margin_px, 1.5)
    assert math.isclose(p.stall_eps_px, 0.15) and p.stall_hist_len == 6


def test_tracker_latch_hysteresis_and_done():
    plan = _plan([(100, 100), (300, 100)], tol=20, min_hold=2, skippable=False)
    tr = anchored_from_plan(plan, start_px=(0.0, 0.0))
    assert tr.n == 3 and tr.i == 0 and not tr.skippable[0] and not tr.skippable[-1]
    tr.update(np.array([[0.0, 0.0]]))            # anchor latches after min_hold 1
    assert tr.i == 1
    tr.update(np.array([[105.0, 100.0]]))        # within tol: hold 1
    assert tr.i == 1 and tr.hold == 1
    tr.update(np.array([[122.0, 100.0]]))        # dead band (20 < 22 < 25): hold kept
    assert tr.hold == 1
    tr.update(np.array([[130.0, 100.0]]))        # beyond 1.25*tol: hold reset
    assert tr.hold == 0
    tr.update(np.array([[100.0, 100.0]])); tr.update(np.array([[100.0, 100.0]]))
    assert tr.i == 2
    out = None
    for _ in range(3):
        out = tr.update(np.array([[300.0, 100.0]]))
    assert tr.done and out['done'] and out['score'] == 1.0
    assert targets_vector(tr)[0] == 0


def test_tracker_ahead_skip():
    # active wp far away, next wp reached: ahead_skip after skip_frames (6) frames
    plan = _plan([(100, 100), (500, 100), (600, 100)], tol=30, min_hold=1)
    tr = anchored_from_plan(plan, start_px=(0.0, 0.0))
    tr.update(np.array([[0.0, 0.0]]))            # latch anchor
    assert tr.i == 1
    # sitting on wp2: transit_skip fires after transit_skip_frames (3), then wp2 latches
    reasons = [tr.update(np.array([[500.0, 100.0]]))['skip_reason'] for _ in range(6)]
    assert 'transit_skip' in reasons and tr.i >= 3


def test_targets_vector_encoding():
    plan = _plan([(10, 20), (30, 40), (50, 60), (70, 80)], tol=55)
    tr = anchored_from_plan(plan, start_px=(1.0, 2.0))
    v = targets_vector(tr, k=3)
    assert v.shape == (8,) and v[0] == 3 and v[1] == 55  # anchor takes wp0's tol
    up, tol = decode_targets(v)
    assert np.allclose(up, [[1, 2], [10, 20], [30, 40]]) and tol == 55
    tr.update(np.array([[1.0, 2.0]])); tr.update(np.array([[10.0, 20.0]]))
    tr.update(np.array([[30.0, 40.0]]))
    v = targets_vector(tr, k=3)
    assert v[0] == 2 and np.allclose(decode_targets(v)[0], [[50, 60], [70, 80]])
    assert np.allclose(targets_vector(None), EMPTY_TARGETS)


def test_snapshot_store_matches_live_episode_tracker():
    cam = _camera()
    xyz = np.stack([np.linspace(0, 0.3, 12), np.zeros(12), np.zeros(12)], axis=1)
    # tol must exceed the ~16 px per-step motion, as upstream's 60 px does at 640 wide:
    # a tolerance below one step can never latch (hold resets every frame)
    plan = _plan([tuple(cam.project(xyz[4])), tuple(cam.project(xyz[11]))], tol=40, min_hold=1)
    ends = np.array([12])
    store = SimplerTrackerSnapshotStore(xyz, ends, {0: plan}, cam)
    live = SimplerEpisodeTracker(plan, cam, xyz[0])
    assert np.allclose(store.targets[0], live.targets_vector())   # anchor row: zero updates
    for f in range(1, 12):
        live.update(xyz[f])
        assert np.allclose(store.targets[f], live.targets_vector()), f
    assert store.final_progress[0] == 1.0 and live.done


# ------------------------------------------------------------------ verifier

def _upstream_cost(pix, upcoming, tol, alpha):
    """Upstream select_best_action_chunk Veritas-px loop, verbatim in numpy."""
    if not np.all(np.isfinite(pix)):
        return float('inf')
    target_idx = 0
    while target_idx < len(upcoming) and np.linalg.norm(pix - upcoming[target_idx]) < tol:
        target_idx += 1
    if target_idx >= len(upcoming):
        return 0.0
    total, w = 0.0, 1.0
    for k in range(target_idx, len(upcoming)):
        total += w * float(np.linalg.norm(pix - upcoming[k]))
        w *= alpha
    return total


def _obs(cam, tcp, targets, B):
    return {'end_effector_pose': torch.tensor(np.tile(np.r_[tcp, 0, 0, 0, 1], (B, 1, 1)), dtype=torch.float32),
            'veritas_targets': torch.tensor(np.tile(targets, (B, 1, 1)), dtype=torch.float32)}


def test_verifier_matches_upstream_loop_on_random_inputs(tmp_path):
    cam, path = _camera(tmp_path)
    rng = np.random.default_rng(0)
    for scale in (None, (0.72, 0.74, 0.87)):
        ver = VeritasPixelVerifier(str(path), action_pos_scale=scale, health_check=False)
        for _ in range(20):
            tcp = rng.uniform(-0.1, 0.1, 3)
            plan = _plan([tuple(rng.uniform([100, 100], [540, 380])) for _ in range(4)],
                         tol=float(rng.uniform(20, 80)))
            tr = anchored_from_plan(plan, cam.project(tcp))
            tr.update(cam.project(tcp)[None])   # latch the anchor
            targets = targets_vector(tr)
            up, tol = decode_targets(targets)
            B = 8
            action = torch.tensor(rng.normal(0, 0.03, (B, 8, 7)), dtype=torch.float32)
            got = -ver.get_value(_obs(cam, tcp, targets, B), action).numpy()
            s = np.ones(3) if scale is None else np.array(scale)
            for b in range(B):
                end = tcp + cam.R_world_from_base @ (action[b, :, :3].numpy() * s).sum(0)
                want = _upstream_cost(cam.project(end), list(up), tol, 0.5)
                assert math.isclose(got[b], want, rel_tol=1e-4, abs_tol=1e-3), (got[b], want)


def test_verifier_all_satisfied_zero_done_zero_and_horizon(tmp_path):
    cam, path = _camera(tmp_path)
    ver = VeritasPixelVerifier(str(path), health_check=False, horizon=4)
    tcp = np.zeros(3)
    plan = _plan([(320, 240), (322, 241)], tol=60)
    tr = anchored_from_plan(plan, cam.project(tcp))
    targets = targets_vector(tr)
    zero = torch.zeros(2, 8, 7)
    assert torch.allclose(ver.get_value(_obs(cam, tcp, targets, 2), zero), torch.zeros(2))
    # only the first `horizon` actions are integrated
    a = torch.zeros(1, 8, 7); a[0, 6, 0] = 1.0     # step 7 is ignored with horizon=4
    assert torch.allclose(ver.get_value(_obs(cam, tcp, targets, 1), a), torch.zeros(1))
    a[0, 0, 0] = 1.0                                 # step 0 counts -> huge move -> nonzero cost
    assert ver.get_value(_obs(cam, tcp, targets, 1), a)[0] < 0
    # done tracker -> value 0 for anything
    assert torch.allclose(ver.get_value(_obs(cam, tcp, EMPTY_TARGETS, 1), a), torch.zeros(1))


def test_verifier_plan_done_falls_back_to_source_object(tmp_path):
    """Upstream: plan done -> pixel distance from the rounded, clipped endpoint to the
    source object (compute_distance_to_target); waypoints still pending ignore it."""
    cam, path = _camera(tmp_path)
    ver = VeritasPixelVerifier(str(path), health_check=False,
                              plan_done_mode='source_object_legacy')
    tcp = np.zeros(3)
    src = np.array([400.0, 300.0])
    done = np.r_[EMPTY_TARGETS, src].astype(np.float32)
    a = torch.zeros(2, 8, 7); a[0, :, 0] = 0.01; a[1, :, 0] = -0.01
    got = -ver.get_value(_obs(cam, tcp, done, 2), a).numpy()
    for b in range(2):
        end = tcp + cam.R_world_from_base @ a[b, :, :3].numpy().sum(0)
        pe = np.clip(np.round(cam.project(end)), 0, [639, 479])
        assert math.isclose(got[b], np.linalg.norm(pe - src), rel_tol=1e-4, abs_tol=1e-3)
    # unknown source pixel -> still 0, as without the slots
    unknown = np.r_[EMPTY_TARGETS, -1.0, -1.0].astype(np.float32)
    assert torch.allclose(ver.get_value(_obs(cam, tcp, unknown, 2), a), torch.zeros(2))
    # pending waypoints: the appended source pixel does not change the cost
    tr = anchored_from_plan(_plan([(100, 100), (120, 110)], tol=10), cam.project(tcp))
    t8 = targets_vector(tr)
    assert torch.allclose(ver.get_value(_obs(cam, tcp, t8, 2), a),
                          ver.get_value(_obs(cam, tcp, np.r_[t8, src].astype(np.float32), 2), a))


def test_completed_plan_training_and_live_scores_match(tmp_path):
    cam, path = _camera(tmp_path)
    ver = VeritasPixelVerifier(str(path), health_check=False,
                              normalize='fixed', value_mean=-20, value_std=10)
    actions = torch.zeros(2, 8, 7)
    actions[0, :, 0], actions[1, :, 0] = 0.01, -0.01
    train = _obs(cam, np.zeros(3), EMPTY_TARGETS, 2)
    live = _obs(cam, np.zeros(3), np.r_[EMPTY_TARGETS, 400., 300.], 2)
    assert torch.equal(ver.get_value(train, actions), ver.get_value(live, actions))
    assert torch.equal(ver.get_value(live, actions), torch.full((2,), 2.0))


def test_executed_horizon_does_not_reward_unexecuted_recovery(tmp_path):
    cam, path = _camera(tmp_path)
    ver = VeritasPixelVerifier(str(path), horizon=4, health_check=False)
    targets = np.array([1, 5, 320, 240, -1, -1, -1, -1], dtype=np.float32)
    actions = torch.zeros(2, 8, 7)
    # A moves away then returns in the unexecuted tail. B stays then moves away.
    actions[0, :4, 0], actions[0, 4:, 0] = 0.02, -0.02
    actions[1, 4:, 0] = 0.02
    obs = _obs(cam, np.zeros(3), targets, 2)
    assert ver.get_value(obs, actions).argmax().item() == 1
    ver.horizon = 8
    assert ver.get_value(obs, actions).argmax().item() == 0


def test_verifier_ranks_toward_waypoint(tmp_path):
    cam, path = _camera(tmp_path)
    ver = VeritasPixelVerifier(str(path), health_check=False)
    tcp = np.zeros(3)
    # wp2 lies beyond wp1 along the same direction (the lookahead cost scores a
    # candidate that satisfies wp1 against wp2, so wp2 must not sit behind it)
    plan = _plan([tuple(cam.project(np.array([-0.2, 0.0, 0.0]))),
                  tuple(cam.project(np.array([-0.4, 0.0, 0.0])))], tol=10)
    tr = anchored_from_plan(plan, cam.project(tcp)); tr.update(cam.project(tcp)[None])
    targets = targets_vector(tr)
    # base-frame +x is world -x under the 180-degree yaw: candidate 0 moves toward wp1
    a = torch.zeros(2, 8, 7); a[0, :, 0] = 0.025; a[1, :, 0] = -0.025
    v = ver.get_value(_obs(cam, tcp, targets, 2), a)
    assert v[0] > v[1]


# ------------------------------------------------------------------ plan store

def test_plan_store_roundtrip_and_provider(tmp_path):
    plan = _plan([(1, 2), (3, 4)])
    rec = tmp_path / 'ep7.json'
    assert not has_plan(rec)
    write_plan_record(rec, None, seed=5)
    assert not has_plan(rec)
    with pytest.raises(ValueError):
        load_plan(rec)
    write_plan_record(rec, plan, usage={'prompt_token_count': 1})
    assert has_plan(rec) and load_plan(rec).waypoints[1].uv == (3.0, 4.0)
    link_seed(tmp_path, 5, 'ep7.json')
    prov = CachedPlanProvider(tmp_path)
    assert prov(5).waypoints[0].uv == (1.0, 2.0) and prov.n_hits == 1
    with pytest.raises(FileNotFoundError):
        prov(6)
    live = CachedPlanProvider(tmp_path, planner=lambda seed, **kw: {'plan': plan.model_dump()},
                              allow_live=True)
    assert live(6).waypoints and has_plan(tmp_path / 'seed6.json') and live.n_live == 1


# ------------------------------------------------------------------ split manifests

def test_split_manifest_rules_nesting_and_loader(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT.parent / 'scriptsv2' / 'splits'))
    import make_simpler_splits as ms
    from diffusion_policy.common.split_manifest import load_split_manifest
    rng = np.random.default_rng(1)
    n = 800
    cells = [(round(float(x), 3), round(float(y), 3)) for x, y in
             zip(rng.choice(np.linspace(-0.13, -0.09, 8), n), rng.choice(np.linspace(0.18, 0.25, 7), n))]
    rows = {i: dict(src_shard='state1.zarr', src_episode=i, seed=200000 + i, seed_gate=True,
                    start_xy=list(cells[i]), start_cell=list(cells[i]), goal_xy=[0, 0], goal_cell=[0, 0])
            for i in range(n)}
    rows[3]['seed_gate'] = False
    ends = np.arange(10, 10 * (n + 1), 10)
    monkeypatch.setattr(ms, 'episode_table', lambda task: ('z.zarr', ends, rows))
    monkeypatch.setitem(ms.TASKS['eggplant'], 'train_sizes',
                        {'random': (150, 50, 20), 'quadrant': (50, 20), 'inner': (50, 20)})
    # the synthetic grid is coarser than the real 28-cell one: widen the inner disc
    monkeypatch.setitem(ms.TASKS['eggplant'], 'inner', dict(center=(-0.1135, 0.2140), radius=0.02))
    for rule in ('random', 'quadrant', 'inner'):
        out = ms.build('eggplant', rule, seed=0)
        prev = None
        for size, m in sorted(out.items()):
            assert len(m['test']) == 50 and len(m['val']) == 20 and len(m['train']) == size
            assert not (set(m['train']) & set(m['test'])) and not (set(m['val']) & set(m['test']))
            assert 3 not in m['test']                              # gate-failed never in test
            if rule != 'random':
                region = set(ms.test_pool('eggplant', rule, rows))
                assert set(m['test']) <= region and not (set(m['train']) & region)
            if prev is not None:
                assert set(prev) <= set(m['train'])                # nested prefixes
            prev = m['train']
            path = tmp_path / f'{rule}_{size}.json'
            path.write_text(json.dumps(m))
            loaded = load_split_manifest(path, episode_ends=ends,
                                         expected_counts={'train': size, 'val': 20, 'test': 50})
            assert loaded['checksum'] == m['checksum']


# ------------------------------------------------------------------ split-only datasets

@pytest.mark.parametrize('task', ['eggplant', 'stack'])
def test_remapped_manifests_match_split_dataset_and_seeds(task):
    """Remapped manifests load against the split-only zarr and name the same seeds (per
    split) as the pre-remap manifests did against the strict zarr."""
    import zarr
    from diffusion_policy.common.split_manifest import load_split_manifest
    split_dir = ROOT / 'diffusion_policy' / 'config' / 'splits' / 'simpler' / task
    backup = split_dir / '_pre_split_dataset'
    if not backup.exists():
        pytest.skip('split-only dataset not built yet')
    for path in sorted(split_dir.glob('*.json')):
        new, old = json.loads(path.read_text()), json.loads((backup / path.name).read_text())
        ends = zarr.open(new['zarr_path'], 'r')['meta/episode_ends'][:]
        load_split_manifest(path, episode_ends=ends)
        for s in ('train', 'val', 'test'):
            seeds_new = sorted(new['episodes'][str(i)]['seed'] for i in new[s])
            seeds_old = sorted(old['episodes'][str(i)]['seed'] for i in old[s])
            assert seeds_new == seeds_old, (path.name, s)
