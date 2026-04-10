"""3D bin-packing solver using Genetic Algorithm with GPU-parallel decoding.

Chromosome = permutation of box indices + rotation flags.
Decoder = heightmap-based greedy bottom-left placement.
Population decoded in parallel: at each step (placing one box per chromosome),
all chromosomes are processed simultaneously on GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from fill3d.cpsat_model import Fill3DInstance, Fill3DPlacement


GRID_RES = 50  # mm per cell


@dataclass
class GA3DConfig:
    pop_size: int = 2048
    elite_ratio: float = 0.1
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_k: int = 4
    time_limit_seconds: float = 60.0
    grid_res: int = GRID_RES
    min_support_ratio: float = 0.8  # reject if support area < this ratio
    seed: int | None = None


@dataclass
class GA3DResult:
    status: str
    packed_count: int
    total_items: int
    packed_volume: int
    pallet_volume: int
    volume_ratio: float
    placements: list[Fill3DPlacement]
    generations: int
    best_fitness_history: list[int]


def solve_ga3d(
    instance: Fill3DInstance,
    config: GA3DConfig | None = None,
) -> GA3DResult:
    """Solve 3D bin packing with GPU-accelerated genetic algorithm."""
    if config is None:
        config = GA3DConfig()

    PL = instance.pallet_length
    PW = instance.pallet_width
    PH = instance.pallet_max_height
    items = instance.items
    n = len(items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.seed is not None:
        torch.manual_seed(config.seed)
    gr = config.grid_res
    gx_count = PL // gr
    gy_count = PW // gr
    pop = config.pop_size

    # Box data
    lengths = torch.tensor([it.length for it in items],
                           dtype=torch.int32, device=device)
    widths = torch.tensor([it.width for it in items],
                          dtype=torch.int32, device=device)
    heights = torch.tensor([it.height for it in items],
                           dtype=torch.int32, device=device)
    volumes = torch.tensor([it.volume for it in items],
                           dtype=torch.int64, device=device)
    can_rotate = torch.tensor(
        [it.allow_rotation and it.length != it.width for it in items],
        dtype=torch.bool, device=device,
    )

    # Precompute unique (gcx, gcy) footprint sizes to batch position search
    # For each item, compute grid footprint for rot=0 and rot=1
    gcx_norot = lengths // gr  # (n,)
    gcy_norot = widths // gr
    gcx_rot = widths // gr
    gcy_rot = lengths // gr

    # --- Initialize population ---
    # Half seeded with height-grouped order (+ shuffle within group),
    # half fully random for diversity.
    height_sorted = heights.argsort(descending=True)  # tallest first
    unique_h = heights.unique()
    n_seeded = pop // 2
    seeded_perms = []
    for _ in range(n_seeded):
        perm = height_sorted.clone()
        for h in unique_h:
            mask = heights[perm] == h
            idx = mask.nonzero(as_tuple=True)[0]
            shuf = idx[torch.randperm(idx.numel(), device=device)]
            perm[idx] = perm[shuf]
        seeded_perms.append(perm)
    random_perms = [
        torch.randperm(n, device=device)
        for _ in range(pop - n_seeded)
    ]
    perms = torch.stack(seeded_perms + random_perms)
    rots = torch.zeros((pop, n), dtype=torch.int32, device=device)
    # Random rotations for rotatable items
    rot_mask = can_rotate.unsqueeze(0).expand(pop, -1)
    rots = torch.where(
        rot_mask,
        torch.randint(0, 2, (pop, n), dtype=torch.int32, device=device),
        rots,
    )

    best_fitness = -1
    best_solution = None
    fitness_history: list[int] = []
    gen = 0
    t0 = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= config.time_limit_seconds:
            break

        # --- Decode ---
        fitness, packed_counts, all_place, all_placed = _decode(
            perms, rots, lengths, widths, heights, volumes,
            gcx_norot, gcy_norot, gcx_rot, gcy_rot,
            PL, PW, PH, gr, gx_count, gy_count, pop, n, device,
            min_support_ratio=config.min_support_ratio,
        )

        # Track best
        max_val, max_idx = fitness.max(dim=0)
        mv = max_val.item()
        if mv > best_fitness:
            best_fitness = mv
            mi = max_idx.item()
            best_solution = (
                perms[mi].clone(),
                rots[mi].clone(),
                all_place[mi].clone(),
                all_placed[mi].clone(),
            )

        fitness_history.append(best_fitness)
        gen += 1

        # --- Evolve ---
        perms, rots = _evolve(
            perms, rots, fitness, can_rotate, config, n, pop, device,
        )

    # --- Extract best ---
    placements: list[Fill3DPlacement] = []
    packed_volume = 0
    if best_solution is not None:
        b_perm, b_rot, b_place, b_placed = best_solution
        for i in range(n):
            if not b_placed[i]:
                continue
            px = b_place[i, 0].item()
            py = b_place[i, 1].item()
            pz = b_place[i, 2].item()
            sx = b_place[i, 3].item()
            sy = b_place[i, 4].item()
            sz = b_place[i, 5].item()
            item = items[i]
            placements.append(Fill3DPlacement(
                item_id=item.item_id, x=px, y=py, z=pz,
                rotated=(sx != item.length or sy != item.width),
                size_x=sx, size_y=sy, size_z=sz,
            ))
            packed_volume += sx * sy * sz

    placements.sort(key=lambda p: (p.z, p.y, p.x, p.item_id))
    max_top_z = max((p.z + p.size_z for p in placements), default=0)
    used_volume = PL * PW * max_top_z
    return GA3DResult(
        status=f"ga_gen{gen}",
        packed_count=len(placements),
        total_items=n,
        packed_volume=packed_volume,
        pallet_volume=used_volume,
        volume_ratio=packed_volume / used_volume if used_volume > 0 else 0.0,
        placements=placements,
        generations=gen,
        best_fitness_history=fitness_history,
    )


def _decode(
    perms, rots, lengths, widths, heights, volumes,
    gcx_norot, gcy_norot, gcx_rot, gcy_rot,
    PL, PW, PH, gr, gx_count, gy_count, pop, n, device,
    *, min_support_ratio: float = 0.8,
):
    """Decode population — no per-footprint loop.

    Grid is small (e.g. 12×10), so we precompute a padded
    cumulative-sum heightmap and evaluate ALL positions for ALL
    chromosomes in one shot per candidate position using the
    2D prefix-sum trick for variable-size window max.

    Approach: iterate over (gx, gy) positions (120 total).
    For each position, compute max height under each
    chromosome's footprint using a pre-built max lookup table
    indexed by (gx, gy, fw, fh).  Since footprints are small
    and grid is tiny, we precompute max_pool2d for ALL kernel
    sizes once per step (14 sizes), store results in a dict,
    then for each (gx, gy) do a single gather.

    This replaces 14 per-footprint loops with 1 batch of 14
    pool calls + 120 position evaluations (all vectorized
    over pop).
    """
    import torch.nn.functional as F

    hmap = torch.zeros((pop, gx_count, gy_count),
                       dtype=torch.float32, device=device)
    fitness = torch.zeros(pop, dtype=torch.int64, device=device)
    packed_counts = torch.zeros(pop, dtype=torch.int32,
                                device=device)
    place_data = torch.zeros((pop, n, 6), dtype=torch.int32,
                             device=device)
    placed = torch.zeros((pop, n), dtype=torch.bool,
                         device=device)

    # Precompute unique footprints → index mapping
    all_gcx = torch.cat([gcx_norot, gcx_rot])
    all_gcy = torch.cat([gcy_norot, gcy_rot])
    all_feet = torch.stack(
        [all_gcx, all_gcy], dim=1,
    ).unique(dim=0)
    valid_feet = [
        (f[0].item(), f[1].item()) for f in all_feet
        if 0 < f[0].item() <= gx_count
        and 0 < f[1].item() <= gy_count
    ]

    # Map (fw, fh) → index for fast lookup
    foot_to_idx = {ft: i for i, ft in enumerate(valid_feet)}
    n_ft = len(valid_feet)

    PENALTY_W = 2.0
    CONTACT_W = 200.0  # reward for side contact (walls + boxes)

    for step in range(n):
        box_ids = perms[:, step]
        rot_flags = rots[:, step]
        bh = heights[box_ids]
        bv = volumes[box_ids]

        is_rot = rot_flags == 1
        gcx = torch.where(
            is_rot, gcx_rot[box_ids], gcx_norot[box_ids],
        )
        gcy = torch.where(
            is_rot, gcy_rot[box_ids], gcy_norot[box_ids],
        )
        sx = gcx * gr
        sy = gcy * gr

        # Per-chromosome footprint index
        ft_idx = torch.zeros(pop, dtype=torch.long,
                             device=device)
        for ft, idx in foot_to_idx.items():
            m = (gcx == ft[0]) & (gcy == ft[1])
            ft_idx[m] = idx

        # Compute max_pool and avg_pool for all footprint
        # sizes at once, store in (n_ft, pop, nx, ny) tensors.
        # Pad results to max (nx, ny) for uniform indexing.
        max_nx = gx_count  # upper bound
        max_ny = gy_count
        # score_map[ft_i, pop_j, gx, gy] = placement score
        score_map = torch.full(
            (n_ft, pop, max_nx, max_ny), float('inf'),
            dtype=torch.float32, device=device,
        )
        zbot_map = torch.full(
            (n_ft, pop, max_nx, max_ny), float(PH + 1),
            dtype=torch.float32, device=device,
        )

        sub = hmap.unsqueeze(1)  # (pop, 1, gx, gy)
        # Occupancy map for contact scoring: pad with 1 (walls)
        occ = (sub > 0).float()
        occ_padded = F.pad(occ, (1, 1, 1, 1), mode='constant', value=1.0)
        for fi, (fw, fh) in enumerate(valid_feet):
            mp = F.max_pool2d(
                sub, kernel_size=(fw, fh), stride=1,
            ).squeeze(1)  # (pop, nx_i, ny_i)
            ap = F.avg_pool2d(
                sub, kernel_size=(fw, fh), stride=1,
            ).squeeze(1)
            # Side contact: count occupied cells in 1-cell
            # border around footprint (walls count as occupied)
            big_area = (fw + 2) * (fh + 2)
            inner_area = fw * fh
            border_cells = big_area - inner_area
            big_sum = F.avg_pool2d(
                occ_padded, kernel_size=(fw + 2, fh + 2), stride=1,
            ).squeeze(1) * big_area
            inner_sum = F.avg_pool2d(
                occ, kernel_size=(fw, fh), stride=1,
            ).squeeze(1) * inner_area
            contact_ratio = (big_sum - inner_sum) / border_cells

            penalty = mp - ap
            sc = mp + penalty * PENALTY_W - contact_ratio * CONTACT_W
            # Reject positions where support ratio < threshold
            ratio = torch.where(
                mp > 0, ap / mp, torch.ones_like(mp),
            )
            unsupported = ratio < min_support_ratio
            sc = torch.where(
                unsupported, torch.full_like(sc, float('inf')), sc,
            )
            _, nxi, nyi = mp.shape
            score_map[fi, :, :nxi, :nyi] = sc
            zbot_map[fi, :, :nxi, :nyi] = torch.where(
                unsupported, torch.full_like(mp, float(PH + 1)), mp,
            )

        # For each chromosome, gather its footprint's score map
        # ft_idx: (pop,) → index into score_map dim 0
        # Gather: score_map[ft_idx[p], p, :, :] for each p
        fi_exp = ft_idx.view(1, pop, 1, 1).expand(
            1, pop, max_nx, max_ny,
        )
        my_scores = score_map.gather(0, fi_exp).squeeze(0)
        my_zbots = zbot_map.gather(0, fi_exp).squeeze(0)
        # (pop, max_nx, max_ny)

        # Find best position per chromosome
        flat_s = my_scores.reshape(pop, -1)
        flat_z = my_zbots.reshape(pop, -1)
        min_scores, min_idxs = flat_s.min(dim=1)
        best_gx = (min_idxs // max_ny).to(torch.int32)
        best_gy = (min_idxs % max_ny).to(torch.int32)
        best_z = flat_z[
            torch.arange(pop, device=device), min_idxs,
        ].to(torch.int32)

        top = best_z + bh
        did_place = top <= PH
        if not did_place.any():
            continue

        new_top = (best_z + bh).float()

        # Heightmap update — precompute all cell offsets,
        # build flat indices, scatter in one shot
        dp_idx = did_place.nonzero(as_tuple=True)[0]
        if dp_idx.numel() > 0:
            dp_gx = best_gx[dp_idx].long()
            dp_gy = best_gy[dp_idx].long()
            dp_gcx = gcx[dp_idx]
            dp_gcy = gcy[dp_idx]
            dp_top = new_top[dp_idx]
            mfw = dp_gcx.max().item()
            mfh = dp_gcy.max().item()
            # Build (dx, dy) offset grid
            dx_grid = torch.arange(
                mfw, device=device,
            ).unsqueeze(0)  # (1, mfw)
            dy_grid = torch.arange(
                mfh, device=device,
            ).unsqueeze(0)  # (1, mfh)
            # Mask: (k, mfw) and (k, mfh)
            dx_mask = dx_grid < dp_gcx.unsqueeze(1)  # (k,mfw)
            dy_mask = dy_grid < dp_gcy.unsqueeze(1)  # (k,mfh)
            # Combined mask: (k, mfw, mfh)
            cell_mask = (
                dx_mask.unsqueeze(2) & dy_mask.unsqueeze(1)
            )
            # Cell coords: (k, mfw, mfh)
            cx = dp_gx.unsqueeze(1).unsqueeze(2) + dx_grid.unsqueeze(2)
            cy = dp_gy.unsqueeze(1).unsqueeze(2) + dy_grid.unsqueeze(1)
            # Flat index into hmap (pop, gx*gy)
            hmap_flat = hmap.reshape(pop, -1)
            flat_cell = cx * gy_count + cy  # (k, mfw, mfh)
            # Expand dp_idx for scatter
            k = dp_idx.numel()
            pi = dp_idx.unsqueeze(1).unsqueeze(2).expand(
                k, mfw, mfh,
            )
            top_val = dp_top.unsqueeze(1).unsqueeze(2).expand(
                k, mfw, mfh,
            )
            # Apply mask and scatter
            m = cell_mask
            hmap_flat[pi[m], flat_cell[m]] = top_val[m]
            hmap = hmap_flat.reshape(pop, gx_count, gy_count)

        # Record
        bi = box_ids[dp_idx]
        px = best_gx[dp_idx] * gr
        py = best_gy[dp_idx] * gr
        place_data[dp_idx, bi, 0] = px
        place_data[dp_idx, bi, 1] = py
        place_data[dp_idx, bi, 2] = best_z[dp_idx]
        place_data[dp_idx, bi, 3] = sx[dp_idx]
        place_data[dp_idx, bi, 4] = sy[dp_idx]
        place_data[dp_idx, bi, 5] = bh[dp_idx]
        placed[dp_idx, bi] = True

        fitness += torch.where(
            did_place, bv.to(torch.int64),
            torch.zeros(1, dtype=torch.int64, device=device),
        )
        packed_counts += did_place.int()

    # Fitness = pack more first, then minimize height (maximize compactness)
    # Primary: packed_count * 1M
    # Secondary: volume_ratio * 10000  (0..10000)
    max_top_z = hmap.amax(dim=(1, 2)).to(torch.int64)  # (pop,)
    pallet_area = PL * PW
    compactness = torch.where(
        max_top_z > 0,
        fitness * 10000 // (pallet_area * max_top_z),
        torch.zeros_like(fitness),
    )
    fitness = packed_counts.to(torch.int64) * 1_000_000 + compactness

    return fitness, packed_counts, place_data, placed


def _evolve(perms, rots, fitness, can_rotate, config, n, pop, device):
    """Selection + crossover + mutation — fully vectorized."""
    n_elite = max(1, int(pop * config.elite_ratio))
    sorted_idx = fitness.argsort(descending=True)

    new_perms = torch.empty_like(perms)
    new_rots = torch.empty_like(rots)

    # Elitism
    new_perms[:n_elite] = perms[sorted_idx[:n_elite]]
    new_rots[:n_elite] = rots[sorted_idx[:n_elite]]

    n_off = pop - n_elite
    # Vectorized tournament selection
    t_idx = torch.randint(0, pop, (n_off, 2, config.tournament_k),
                          device=device)
    t_fit = fitness[t_idx]
    t_best = t_fit.argmax(dim=2)
    arange_off = torch.arange(n_off, device=device)
    p1_idx = t_idx[arange_off, 0, t_best[:, 0]]
    p2_idx = t_idx[arange_off, 1, t_best[:, 1]]

    p1_perms = perms[p1_idx]  # (n_off, n)
    p2_perms = perms[p2_idx]
    p1_rots = rots[p1_idx]

    # Batch crossover: use random segment swap + repair
    do_cx = torch.rand(n_off, device=device) < config.crossover_rate
    children = _batch_crossover(
        p1_perms, p2_perms, do_cx, n, n_off, device,
    )

    # Batch swap mutation
    do_mut = torch.rand(n_off, device=device) < config.mutation_rate
    mut_idx = do_mut.nonzero(as_tuple=True)[0]
    if mut_idx.numel() > 0:
        a = torch.randint(0, n, (mut_idx.numel(),), device=device)
        b = torch.randint(0, n, (mut_idx.numel(),), device=device)
        va = children[mut_idx, a].clone()
        children[mut_idx, a] = children[mut_idx, b]
        children[mut_idx, b] = va

    new_perms[n_elite:] = children

    # Rotation: copy from p1, mutate some
    child_rots = p1_rots.clone()
    do_rmut = torch.rand(n_off, device=device) < config.mutation_rate
    rm_idx = do_rmut.nonzero(as_tuple=True)[0]
    if rm_idx.numel() > 0:
        j = torch.randint(0, n, (rm_idx.numel(),), device=device)
        box_j = children[rm_idx, j]
        rotatable = can_rotate[box_j]
        flip_idx = rm_idx[rotatable]
        flip_j = j[rotatable]
        if flip_idx.numel() > 0:
            child_rots[flip_idx, flip_j] = (
                1 - child_rots[flip_idx, flip_j]
            )
    new_rots[n_elite:] = child_rots

    return new_perms, new_rots


def _batch_crossover(p1, p2, do_cx, n, batch, device):
    """Batch-friendly crossover using position-based approach.

    For each pair where do_cx is True:
    - Pick random segment from p1
    - Fill rest from p2 preserving order
    Uses scatter/gather for GPU efficiency.
    """
    # Start with p1 copy for non-crossover individuals
    children = p1.clone()

    cx_idx = do_cx.nonzero(as_tuple=True)[0]
    if cx_idx.numel() == 0:
        return children

    k = cx_idx.numel()
    # Random segment [a, b] for each crossover
    ab = torch.randint(0, n, (k, 2), device=device)
    a = ab.min(dim=1).values
    b = ab.max(dim=1).values
    b = torch.clamp(b, min=a + 1)

    # For each crossover individual, build child:
    # Segment [a,b] from p1, rest filled from p2 in order
    cx_p1 = p1[cx_idx]  # (k, n)
    cx_p2 = p2[cx_idx]  # (k, n)

    # Build segment mask
    pos = torch.arange(n, device=device).unsqueeze(0)  # (1, n)
    seg_mask = (pos >= a.unsqueeze(1)) & (pos <= b.unsqueeze(1))  # (k, n)

    # Values in segment
    seg_vals = cx_p1.clone()
    seg_vals[~seg_mask] = -1

    # For each individual, mark which values are in segment
    # Create a (k, max_val+1) lookup
    in_seg = torch.zeros((k, n), dtype=torch.bool, device=device)
    for i in range(k):
        in_seg[i] = False
        seg_items = cx_p1[i, a[i]:b[i] + 1]
        in_seg[i, seg_items.long()] = True

    # Filter p2 values not in segment
    p2_not_in_seg = ~in_seg.gather(1, cx_p2.long())  # (k, n)

    # Build children
    result = torch.full((k, n), -1, dtype=p1.dtype, device=device)
    # Copy segment
    result[seg_mask] = cx_p1[seg_mask]

    # Fill remaining positions from p2 in order
    fill_mask = ~seg_mask  # positions to fill
    for i in range(k):
        fill_vals = cx_p2[i][p2_not_in_seg[i]]
        fill_positions = fill_mask[i].nonzero(as_tuple=True)[0]
        result[i, fill_positions] = fill_vals

    children[cx_idx] = result
    return children
