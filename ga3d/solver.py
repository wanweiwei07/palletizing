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


GRID_RES = 10  # mm per cell


@dataclass
class GA3DConfig:
    pop_size: int = 2048
    elite_ratio: float = 0.1
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_k: int = 4
    time_limit_seconds: float = 60.0
    grid_res_xy: int = GRID_RES
    grid_res_z: int = GRID_RES
    min_support_ratio: float = 0.8  # reject if support area < this ratio
    seed: int | None = None
    use_triton: bool = True  # fused per-chromosome scan kernel


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
    gr_xy = config.grid_res_xy
    gr_z = config.grid_res_z
    gx_count = PL // gr_xy
    gy_count = PW // gr_xy
    PH_v = PH // gr_z
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

    # Footprint cells = ceil(dim / gr_xy) so a box smaller than its fp
    # never overlaps a neighbor (conservative). Output uses original dims.
    gcx_norot = (lengths + gr_xy - 1) // gr_xy
    gcy_norot = (widths + gr_xy - 1) // gr_xy
    gcx_rot = (widths + gr_xy - 1) // gr_xy
    gcy_rot = (lengths + gr_xy - 1) // gr_xy

    # --- Initialize population ---
    # 2/3 of pop: sorted long->short by length.
    #   - 1/3 rotates steps [0, n/2); 1/3 rotates steps [n/2, n).
    # Remaining 1/3: random permutation with alternating rotation.
    n_third = pop // 3
    n_sort = 2 * n_third

    length_sorted = lengths.argsort(descending=True).to(torch.int64)
    sorted_part = length_sorted.unsqueeze(0).expand(n_sort, -1).contiguous()
    random_part = torch.stack([
        torch.randperm(n, device=device) for _ in range(pop - n_sort)
    ])
    perms = torch.cat([sorted_part, random_part], dim=0)

    half = n // 2
    step_pattern = torch.zeros((pop, n), dtype=torch.int32, device=device)
    step_pattern[:n_third, :half] = 1       # group 1: first half steps
    step_pattern[n_third:n_sort, half:] = 1  # group 2: second half steps
    step_pattern[n_sort:, 0::2] = 1          # group 3: alternating

    rots = torch.where(
        can_rotate[perms],
        step_pattern,
        torch.zeros_like(step_pattern),
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
            PL, PW, PH, gr_xy, gr_z, gx_count, gy_count, pop, n, device,
            min_support_ratio=config.min_support_ratio,
            use_triton=config.use_triton,
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
        # b_rot is indexed by step in the permutation, not by box id.
        # Invert b_perm so we can look up each box's placement step.
        step_of_box = torch.empty(n, dtype=torch.long, device=b_perm.device)
        step_of_box[b_perm.long()] = torch.arange(
            n, device=b_perm.device,
        )
        stride_x = gy_count * PH_v
        for i in range(n):
            if not b_placed[i]:
                continue
            cid = b_place[i].item()
            gx_i = cid // stride_x
            gy_i = (cid // PH_v) % gy_count
            gz_i = cid % PH_v
            px = gx_i * gr_xy
            py = gy_i * gr_xy
            pz = gz_i * gr_z
            item = items[i]
            step_i = step_of_box[i].item()
            rotated = bool(b_rot[step_i].item() == 1)
            # Output uses original item dims (not ceil-rounded footprint).
            sx = item.width if rotated else item.length
            sy = item.length if rotated else item.width
            sz = item.height
            placements.append(Fill3DPlacement(
                item_id=item.item_id, x=px, y=py, z=pz,
                rotated=rotated,
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
    PL, PW, PH, gr_xy, gr_z, gx_count, gy_count, pop, n, device,
    *, min_support_ratio: float = 0.8,
    use_triton: bool = True,
):
    """Decode population on GPU in PyTorch.

    hmap stores z-top id (int32) as popcount cache; voxel stores bit-packed
    occupancy. Per step: bucket chromosomes by (fw, fh), pool hmap for
    z_bot/penalty/contact, approximate ap/mp filter, argmin, then a single
    exact-support bit check at the argmin position per chromosome.
    """
    import torch.nn.functional as F

    if use_triton:
        from ga3d.triton_decode import triton_scan

    heights_v = (heights + gr_z - 1) // gr_z
    PH_v = PH // gr_z
    max_dim_mm = max(
        int(lengths.max().item()), int(widths.max().item()),
    )
    max_fp_cells = (max_dim_mm + gr_xy - 1) // gr_xy

    # hmap[p, gx, gy] = z-top id (next-empty voxel layer, 0..PH_v).
    # Acts as popcount cache so z_bot lookup is one load instead of 5
    # voxel reads + bit ops. int32 since value is an integer index.
    hmap = torch.zeros((pop, gx_count, gy_count),
                       dtype=torch.int32, device=device)
    NWORDS = (PH_v + 31) // 32
    voxel = torch.zeros((pop, gx_count, gy_count, NWORDS),
                        dtype=torch.int32, device=device)
    fitness = torch.zeros(pop, dtype=torch.int64, device=device)
    packed_counts = torch.zeros(pop, dtype=torch.int32, device=device)
    # One packed cell-ID per box: id = gx * (GY * PH_v) + gy * PH_v + gz.
    place_data = torch.zeros((pop, n), dtype=torch.int32, device=device)
    placed = torch.zeros((pop, n), dtype=torch.bool, device=device)

    PENALTY_W = 2.0
    CONTACT_W = 500.0
    DIST_W = 1.0 * (gr_xy / 50.0)

    gx_ax = torch.arange(
        gx_count, device=device, dtype=torch.float32,
    ).unsqueeze(1)
    gy_ax = torch.arange(
        gy_count, device=device, dtype=torch.float32,
    ).unsqueeze(0)
    dist_map = (gx_ax + gy_ax) * DIST_W  # (GX, GY), fp32

    key_base = gy_count + 2

    for step in range(n):
        box_ids = perms[:, step]
        rot_flags = rots[:, step]
        bh = heights_v[box_ids]
        bv = volumes[box_ids]

        is_rot = rot_flags == 1
        gcx = torch.where(is_rot, gcx_rot[box_ids], gcx_norot[box_ids])
        gcy = torch.where(is_rot, gcy_rot[box_ids], gcy_norot[box_ids])

        feet_key = gcx.to(torch.int64) * key_base + gcy.to(torch.int64)
        unique_keys, inverse = torch.unique(feet_key, return_inverse=True)

        if use_triton:
            best_gx, best_gy, best_z, best_score = triton_scan(
                hmap.contiguous(),
                gcx.to(torch.int32).contiguous(),
                gcy.to(torch.int32).contiguous(),
                penalty_w=PENALTY_W,
                contact_w=CONTACT_W,
                dist_w=DIST_W,
                gr_z=gr_z,
                min_support_ratio=min_support_ratio,
            )
            # Guard against degenerate fw/fh > pallet (kernel would return
            # INF anyway since pos_mask is empty, but be explicit).
            oob = (gcx <= 0) | (gcy <= 0) | (gcx > gx_count) | (gcy > gy_count)
            if oob.any():
                best_score = torch.where(
                    oob, torch.full_like(best_score, float('inf')), best_score,
                )
        else:
            best_score = torch.full(
                (pop,), float('inf'), dtype=torch.float32, device=device,
            )
            best_gx = torch.zeros(pop, dtype=torch.int32, device=device)
            best_gy = torch.zeros(pop, dtype=torch.int32, device=device)
            best_z = torch.zeros(pop, dtype=torch.int32, device=device)

            for u_idx in range(unique_keys.numel()):
                key = unique_keys[u_idx].item()
                fw = key // key_base
                fh = key % key_base
                if fw <= 0 or fh <= 0 or fw > gx_count or fh > gy_count:
                    continue
                sel = (inverse == u_idx).nonzero(as_tuple=True)[0]
                k_f = sel.numel()
                if k_f == 0:
                    continue

                # Cast subset to fp32 once for F.max_pool2d / F.avg_pool2d
                # (neither accepts int32). Values are exact integers in fp32.
                sub = hmap[sel].float().unsqueeze(1)  # (k_f, 1, GX, GY)
                mp = F.max_pool2d(
                    sub, kernel_size=(fw, fh), stride=1,
                ).squeeze(1)
                ap = F.avg_pool2d(
                    sub, kernel_size=(fw, fh), stride=1,
                ).squeeze(1)

                occ = (sub > 0).float()
                occ_padded = F.pad(
                    occ, (1, 1, 1, 1), mode='constant', value=1.0,
                )
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
                _, nxi, nyi = mp.shape
                sc = (
                    (mp + penalty * PENALTY_W) * gr_z
                    - contact_ratio * CONTACT_W
                    + dist_map[:nxi, :nyi]
                )
                # Cheap approximate support filter (ap/mp), per candidate.
                ratio = torch.where(mp > 0, ap / mp, torch.ones_like(mp))
                sc = torch.where(
                    ratio < min_support_ratio,
                    torch.full_like(sc, float('inf')), sc,
                )

                flat_sc = sc.reshape(k_f, -1)
                flat_mp = mp.reshape(k_f, -1)
                min_s, min_i = flat_sc.min(dim=1)
                gx_i = (min_i // nyi).to(torch.int32)
                gy_i = (min_i % nyi).to(torch.int32)
                z_i = flat_mp.gather(
                    1, min_i.unsqueeze(1),
                ).squeeze(1).to(torch.int32)

                best_score[sel] = min_s
                best_gx[sel] = gx_i
                best_gy[sel] = gy_i
                best_z[sel] = z_i

        # Exact support check hoisted out of per-footprint loop.
        # For each pop's final argmin (best_gx, best_gy, best_z), read
        # voxel bits at z-1 across its own (gcx, gcy) footprint in one
        # vectorized pass. Uses the same max_fp_cells pad + cell_mask
        # pattern as the scatter below.
        finite = torch.isfinite(best_score)
        active = finite & (best_z > 0)
        if active.any():
            ai = active.nonzero(as_tuple=True)[0]
            k_a = ai.numel()
            a_gx = best_gx[ai].long()
            a_gy = best_gy[ai].long()
            a_z = best_z[ai] - 1
            a_fw = gcx[ai]
            a_fh = gcy[ai]
            word_i = (a_z // 32).long()
            bit_i = (a_z - word_i.to(torch.int32) * 32).long()
            mfw = max_fp_cells
            mfh = max_fp_cells
            dxg = torch.arange(mfw, device=device).unsqueeze(0)
            dyg = torch.arange(mfh, device=device).unsqueeze(0)
            fw_mask = dxg < a_fw.unsqueeze(1)   # (k_a, mfw)
            fh_mask = dyg < a_fh.unsqueeze(1)   # (k_a, mfh)
            cell_mask = fw_mask.unsqueeze(2) & fh_mask.unsqueeze(1)
            cx = a_gx.unsqueeze(1).unsqueeze(2) + dxg.unsqueeze(2)
            cy = a_gy.unsqueeze(1).unsqueeze(2) + dyg.unsqueeze(1)
            # Clamp since cells beyond actual footprint would be OOB;
            # cell_mask below zeros out their contribution anyway.
            cx = cx.clamp(0, gx_count - 1)
            cy = cy.clamp(0, gy_count - 1)
            pi = ai.unsqueeze(1).unsqueeze(2).expand(k_a, mfw, mfh)
            wi = word_i.unsqueeze(1).unsqueeze(2).expand(k_a, mfw, mfh)
            w_vals = voxel[pi, cx, cy, wi]
            bi_b = bit_i.unsqueeze(1).unsqueeze(2).to(torch.int32)
            bits = ((w_vals >> bi_b) & 1).to(torch.int32)
            bits = torch.where(cell_mask, bits, torch.zeros_like(bits))
            sup_sum = bits.sum(dim=(1, 2)).to(torch.float32)
            area = (a_fw * a_fh).to(torch.float32)
            sup_ratio = sup_sum / area
            bad = ai[sup_ratio < min_support_ratio]
            if bad.numel() > 0:
                best_score[bad] = float('inf')

        # --- Phase 3 void-fill (z-collapse to 0) ---
        # If the box's footprint at (best_gx, best_gy) has no voxel bits set
        # in [0, fz), the placement can drop to z=0 (overhang/bridge case:
        # hmap reflects upper-box top, but the region below is empty).
        collapse_mask = torch.isfinite(best_score) & (best_z > 0)
        if collapse_mask.any():
            # Per-chromosome word-mask for bits [0, fz) = [0, bh).
            word_ids = torch.arange(NWORDS, device=device).view(1, NWORDS)
            w_lo_vox = word_ids * 32
            w_hi_vox = w_lo_vox + 32
            hi_clip = torch.minimum(bh.unsqueeze(1), w_hi_vox)
            width_low = (hi_clip - w_lo_vox).clamp(min=0, max=32)
            # mask = (1 << width_low) - 1  (built in int64 to cover width=32)
            mask_low = (
                (torch.ones_like(width_low, dtype=torch.int64) << width_low.to(torch.int64))
                - 1
            ).to(torch.int32)  # (pop, NWORDS)

            # col_clear[pid, cx, cy]: True iff voxel bits [0, bh) all zero.
            col_hit = torch.zeros(
                (pop, gx_count, gy_count), dtype=torch.bool, device=device,
            )
            for w in range(NWORDS):
                mw = mask_low[:, w].view(pop, 1, 1)
                col_hit |= (voxel[:, :, :, w] & mw) != 0
            col_clear = ~col_hit

            # Sample col_clear at each chromosome's best-footprint cells.
            mfw = max_fp_cells
            mfh = max_fp_cells
            dxg = torch.arange(mfw, device=device).unsqueeze(0)
            dyg = torch.arange(mfh, device=device).unsqueeze(0)
            fw_m = dxg < gcx.unsqueeze(1)
            fh_m = dyg < gcy.unsqueeze(1)
            cell_m = fw_m.unsqueeze(2) & fh_m.unsqueeze(1)
            cx_q = best_gx.unsqueeze(1).unsqueeze(2).long() + dxg.unsqueeze(2).long()
            cy_q = best_gy.unsqueeze(1).unsqueeze(2).long() + dyg.unsqueeze(1).long()
            cx_q = cx_q.clamp(0, gx_count - 1)
            cy_q = cy_q.clamp(0, gy_count - 1)
            pi_q = (
                torch.arange(pop, device=device).view(-1, 1, 1).expand_as(cx_q)
            )
            fp_clear = col_clear[pi_q, cx_q, cy_q]
            # OOB footprint cells treated as clear.
            fp_clear_ok = fp_clear | ~cell_m
            all_clear = fp_clear_ok.all(dim=(1, 2))

            collapse = collapse_mask & all_clear
            if collapse.any():
                best_z = torch.where(
                    collapse, torch.zeros_like(best_z), best_z,
                )

        top = best_z + bh
        did_place = torch.isfinite(best_score) & (top <= PH_v)
        if not did_place.any():
            continue

        new_top = best_z + bh  # int32, matches hmap dtype

        dp_idx = did_place.nonzero(as_tuple=True)[0]
        dp_gx = best_gx[dp_idx].long()
        dp_gy = best_gy[dp_idx].long()
        dp_gcx = gcx[dp_idx]
        dp_gcy = gcy[dp_idx]
        dp_top = new_top[dp_idx]
        # Use global upper bound to avoid .max().item() sync each step.
        # Cells outside each box's real footprint are masked out below.
        mfw = max_fp_cells
        mfh = max_fp_cells
        dx_grid = torch.arange(mfw, device=device).unsqueeze(0)
        dy_grid = torch.arange(mfh, device=device).unsqueeze(0)
        dx_mask = dx_grid < dp_gcx.unsqueeze(1)
        dy_mask = dy_grid < dp_gcy.unsqueeze(1)
        cell_mask = dx_mask.unsqueeze(2) & dy_mask.unsqueeze(1)
        cx = dp_gx.unsqueeze(1).unsqueeze(2) + dx_grid.unsqueeze(2)
        cy = dp_gy.unsqueeze(1).unsqueeze(2) + dy_grid.unsqueeze(1)
        hmap_flat = hmap.reshape(pop, -1)
        flat_cell = cx * gy_count + cy
        k = dp_idx.numel()
        pi = dp_idx.unsqueeze(1).unsqueeze(2).expand(k, mfw, mfh)
        top_val = dp_top.unsqueeze(1).unsqueeze(2).expand(k, mfw, mfh)
        hmap_flat[pi[cell_mask], flat_cell[cell_mask]] = top_val[cell_mask]
        hmap = hmap_flat.reshape(pop, gx_count, gy_count)

        # Voxel scatter-OR: set bits [z_lo, z_hi) per placed footprint.
        # Work in int64 so (1 << 32) does not overflow during mask build;
        # cast to int32 preserves low-32 bits for OR into voxel.
        z_lo = best_z[dp_idx].to(torch.int64)              # (k,)
        z_hi = z_lo + bh[dp_idx].to(torch.int64)
        voxel_flat = voxel.reshape(pop, gx_count * gy_count, NWORDS)
        for w in range(NWORDS):
            w_lo = w * 32
            w_hi = w_lo + 32
            c_lo = torch.clamp(z_lo, min=w_lo, max=w_hi)
            c_hi = torch.clamp(z_hi, min=w_lo, max=w_hi)
            width = c_hi - c_lo                            # (k,), 0..32
            shift = c_lo - w_lo
            mask64 = torch.where(
                width > 0,
                ((torch.ones_like(width) << width) - 1) << shift,
                torch.zeros_like(width),
            )
            mask32 = mask64.to(torch.int32)                # (k,)
            bw = mask32.unsqueeze(1).unsqueeze(2).expand(k, mfw, mfh)
            voxel_flat[pi[cell_mask], flat_cell[cell_mask], w] |= bw[cell_mask]
        voxel = voxel_flat.reshape(pop, gx_count, gy_count, NWORDS)

        bi = box_ids[dp_idx]
        cell_id = (
            best_gx[dp_idx] * (gy_count * PH_v)
            + best_gy[dp_idx] * PH_v
            + best_z[dp_idx]
        )
        place_data[dp_idx, bi] = cell_id
        placed[dp_idx, bi] = True

        fitness += torch.where(
            did_place, bv.to(torch.int64),
            torch.zeros(1, dtype=torch.int64, device=device),
        )
        packed_counts += did_place.int()

    # Fitness = pack more first, then minimize height (maximize compactness)
    # Primary: packed_count * 1M
    # Secondary: volume_ratio * 10000  (0..10000)
    # hmap is in voxel-z units; rescale to mm to match pallet_area (mm^2).
    max_top_z = hmap.amax(dim=(1, 2)).to(torch.int64) * gr_z  # (pop,), mm
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
