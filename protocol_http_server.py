from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from palletizing import (
    PalletSpec,
    TaskBox,
    plan_multitype_palletizing_3d,
    plan_multitype_palletizing_ga3d,
    plan_multitype_palletizing_ga3d_seam,
)


PROTOCOL_VERSION = "1.5.0"
SOURCE_SYSTEM = "algo"
PALLET_PLAN_SCHEMA_VERSION = "PalletPlan-1.5.0"
ALGO_ENDPOINT = "/v1/algorithms/pallet-plans"

SUPPORTED_ALGORITHMS = frozenset({"ga3d", "ga3d-seam", "3d"})
SUPPORTED_ORIENTATION_CODES = frozenset({"xyz", "yxz"})


class ProtocolError(Exception):
    def __init__(
        self,
        code: int,
        message: str,
        http_status: int = HTTPStatus.BAD_REQUEST,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or []


@dataclass(frozen=True)
class AlgoBox:
    item_id: str
    sku_code: str
    length: float
    width: float
    height: float
    weight: float
    allowed_yaws: tuple[float, ...]


@dataclass(frozen=True)
class ExistingPalletState:
    pallet_id: str
    placements: list[dict[str, Any]]
    load_summary: dict[str, Any]
    is_locked: bool


@dataclass(frozen=True)
class SolveOutcome:
    plan_result: Any
    resolved_algorithm: str
    warning: str | None = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _require_map(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(
            1002,
            f"field type error: {path} should be object",
            details=[{"code": 1002, "path": path, "message": "expected object"}],
        )
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(
            1002,
            f"field type error: {path} should be array",
            details=[{"code": 1002, "path": path, "message": "expected array"}],
        )
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(
            1001 if value is None else 1002,
            f"missing or invalid field: {path}",
            details=[{"code": 1001 if value is None else 1002, "path": path, "message": "expected non-empty string"}],
        )
    return value.strip()


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(
            1002,
            f"field type error: {path} should be boolean",
            details=[{"code": 1002, "path": path, "message": "expected boolean"}],
        )
    return value


def _require_positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ProtocolError(
            1002,
            f"invalid value: {path} should be positive number",
            details=[{"code": 1002, "path": path, "message": "expected positive number"}],
        )
    return float(value)


def _get_dims3d(value: Any, path: str) -> tuple[float, float, float]:
    dims = _require_map(value, path)
    return (
        _require_positive_number(dims.get("length"), f"{path}.length"),
        _require_positive_number(dims.get("width"), f"{path}.width"),
        _require_positive_number(dims.get("height"), f"{path}.height"),
    )


def _validate_constraints(scene_constraints: dict[str, Any]) -> None:
    for key in ("materialKnowledge", "palletPolicy", "bufferPolicy"):
        if key not in scene_constraints:
            raise ProtocolError(
                1001,
                f"missing required field: scene.constraints.{key}",
                details=[{"code": 1001, "path": f"payload.scene.constraints.{key}", "message": "missing required field"}],
            )

    material_knowledge = _require_map(
        scene_constraints["materialKnowledge"],
        "scene.constraints.materialKnowledge",
    )
    pallet_policy = _require_map(
        scene_constraints["palletPolicy"],
        "scene.constraints.palletPolicy",
    )
    buffer_policy = _require_map(
        scene_constraints["bufferPolicy"],
        "scene.constraints.bufferPolicy",
    )

    mode = _require_str(
        material_knowledge.get("mode"),
        "scene.constraints.materialKnowledge.mode",
    )
    _require_bool(
        material_knowledge.get("reorderAllowed"),
        "scene.constraints.materialKnowledge.reorderAllowed",
    )
    if mode == "LOOKAHEAD_N_FIXED_ORDER":
        lookahead = material_knowledge.get("lookaheadCount")
        if not isinstance(lookahead, int) or lookahead < 1:
            raise ProtocolError(
                2001,
                "scene constraint conflict: lookaheadCount is required when materialKnowledge.mode=LOOKAHEAD_N_FIXED_ORDER",
                details=[{"code": 2001, "path": "payload.scene.constraints.materialKnowledge.lookaheadCount", "message": "lookaheadCount must be >= 1"}],
            )

    pallet_mode = _require_str(
        pallet_policy.get("mode"),
        "scene.constraints.palletPolicy.mode",
    )
    if pallet_mode == "FIXED_N_PALLETS":
        pallet_count = pallet_policy.get("palletCount")
        if not isinstance(pallet_count, int) or pallet_count < 1:
            raise ProtocolError(
                2001,
                "scene constraint conflict: palletCount must be >=1 for FIXED_N_PALLETS",
                details=[{"code": 2001, "path": "payload.scene.constraints.palletPolicy.palletCount", "message": "palletCount must be >= 1"}],
            )

    buffer_mode = _require_str(
        buffer_policy.get("mode"),
        "scene.constraints.bufferPolicy.mode",
    )
    if buffer_mode == "FIXED_BUFFER_SLOTS":
        slot_count = buffer_policy.get("slotCount")
        if not isinstance(slot_count, int) or slot_count < 1:
            raise ProtocolError(
                2001,
                "scene constraint conflict: slotCount must be >=1 for FIXED_BUFFER_SLOTS",
                details=[{"code": 2001, "path": "payload.scene.constraints.bufferPolicy.slotCount", "message": "slotCount must be >= 1"}],
            )


def _collect_constraint_warnings(
    payload: dict[str, Any],
    scene_constraints: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    material_knowledge = scene_constraints["materialKnowledge"]
    if material_knowledge.get("mode") != "ALL_KNOWN_REORDERABLE" or not material_knowledge.get("reorderAllowed", False):
        warnings.append(
            "materialKnowledge fixed-order constraints are treated as advisory by the ga3d service."
        )

    buffer_policy = scene_constraints["bufferPolicy"]
    if buffer_policy.get("mode") != "NO_BUFFER":
        warnings.append(
            "bufferPolicy is validated but not actively optimized by the current ga3d service."
        )

    if scene_constraints.get("stabilityLevel") is not None:
        warnings.append(
            "stabilityLevel is applied heuristically rather than as a hard constraint."
        )
    if scene_constraints.get("centerOfGravityLimit") is not None:
        warnings.append(
            "centerOfGravityLimit is not explicitly enforced by the current ga3d service."
        )

    for item in payload.get("materialItems", []):
        if not isinstance(item, dict):
            continue
        if item.get("fragile") is True or item.get("stackable") is False or item.get("maxStackLayers") is not None:
            warnings.append(
                "fragile/stackable/maxStackLayers are accepted as metadata but not strictly enforced."
            )
            break

    return warnings


def _resolve_allowed_yaws(
    item: dict[str, Any],
    idx: int,
) -> tuple[float, ...]:
    orientation_rules = item.get("orientationRules") or {}
    if not isinstance(orientation_rules, dict):
        raise ProtocolError(
            1002,
            f"field type error: materialItems[{idx}].orientationRules should be object",
            details=[{"code": 1002, "path": f"payload.materialItems[{idx}].orientationRules", "message": "expected object"}],
        )

    orientation_mode = orientation_rules.get("orientationMode")
    if orientation_mode is not None and _require_str(
        orientation_mode,
        f"materialItems[{idx}].orientationRules.orientationMode",
    ) != "ORTHOGONAL_ONLY":
        raise ProtocolError(
            1004,
            f"invalid orientation: materialItems[{idx}] only supports ORTHOGONAL_ONLY",
            details=[{"code": 1004, "path": f"payload.materialItems[{idx}].orientationRules.orientationMode", "message": "unsupported orientation mode"}],
        )

    for axis in ("canRotateX", "canRotateY"):
        raw_value = orientation_rules.get(axis)
        if raw_value is None:
            continue
        if _require_bool(raw_value, f"materialItems[{idx}].orientationRules.{axis}"):
            raise ProtocolError(
                1004,
                f"invalid orientation: materialItems[{idx}] cannot rotate around X/Y in ga3d service",
                details=[{"code": 1004, "path": f"payload.materialItems[{idx}].orientationRules.{axis}", "message": "rotation around X/Y is unsupported"}],
            )

    allowed_orientations = orientation_rules.get("allowedOrientations")
    if allowed_orientations is not None:
        raw_codes = _require_list(
            allowed_orientations,
            f"materialItems[{idx}].orientationRules.allowedOrientations",
        )
        codes = {
            _require_str(
                code,
                f"materialItems[{idx}].orientationRules.allowedOrientations[]",
            )
            for code in raw_codes
        }
        if not codes:
            raise ProtocolError(
                1004,
                f"invalid orientation: materialItems[{idx}] allowedOrientations cannot be empty",
            )
        unsupported = sorted(codes - SUPPORTED_ORIENTATION_CODES)
        if unsupported:
            raise ProtocolError(
                1004,
                f"invalid orientation: materialItems[{idx}] contains unsupported upright codes {unsupported}",
                details=[{"code": 1004, "path": f"payload.materialItems[{idx}].orientationRules.allowedOrientations", "message": "only xyz/yxz are supported"}],
            )
        if codes == {"xyz"}:
            return (0.0,)
        if codes == {"yxz"}:
            return (math.pi / 2,)
        return (0.0, math.pi / 2)

    can_rotate_z = orientation_rules.get("canRotateZ")
    if can_rotate_z is None:
        return (0.0, math.pi / 2)
    if _require_bool(can_rotate_z, f"materialItems[{idx}].orientationRules.canRotateZ"):
        return (0.0, math.pi / 2)
    return (0.0,)


def _expand_material_items(
    material_items: list[Any],
    warnings: list[str],
) -> tuple[list[AlgoBox], dict[str, float], dict[str, tuple[float, float, float]]]:
    if not material_items:
        raise ProtocolError(
            1001,
            "missing required field: payload.materialItems",
            details=[{"code": 1001, "path": "payload.materialItems", "message": "materialItems must not be empty"}],
        )

    boxes: list[AlgoBox] = []
    weight_map: dict[str, float] = {}
    original_dims_map: dict[str, tuple[float, float, float]] = {}
    expanded_qty_warning = False

    for idx, raw_item in enumerate(material_items):
        item = _require_map(raw_item, f"materialItems[{idx}]")
        base_item_id = _require_str(item.get("itemId"), f"materialItems[{idx}].itemId")
        sku_code = _require_str(item.get("skuCode"), f"materialItems[{idx}].skuCode")
        qty = item.get("qty")
        if not isinstance(qty, int) or qty <= 0:
            raise ProtocolError(
                1002,
                f"invalid value: materialItems[{idx}].qty must be integer > 0",
                details=[{"code": 1002, "path": f"payload.materialItems[{idx}].qty", "message": "expected integer > 0"}],
            )

        if qty != 1 and not expanded_qty_warning:
            warnings.append(
                "qty != 1 was expanded for compatibility, but AlgoInput v1.5 expects per-item instances."
            )
            expanded_qty_warning = True

        length, width, height = _get_dims3d(
            item.get("dimensions"),
            f"materialItems[{idx}].dimensions",
        )
        weight = _require_positive_number(item.get("weight"), f"materialItems[{idx}].weight")
        allowed_yaws = _resolve_allowed_yaws(item, idx)

        for expansion_index in range(qty):
            instance_item_id = (
                base_item_id
                if qty == 1
                else f"{base_item_id}#{expansion_index + 1}"
            )
            if instance_item_id in weight_map:
                raise ProtocolError(
                    2001,
                    f"duplicate itemId after qty expansion: {instance_item_id}",
                    details=[{"code": 2001, "path": "payload.materialItems", "message": f"duplicate itemId {instance_item_id}"}],
                )

            boxes.append(
                AlgoBox(
                    item_id=instance_item_id,
                    sku_code=sku_code,
                    length=length,
                    width=width,
                    height=height,
                    weight=weight,
                    allowed_yaws=allowed_yaws,
                )
            )
            weight_map[instance_item_id] = weight
            original_dims_map[instance_item_id] = (length, width, height)

    return boxes, weight_map, original_dims_map


def _extract_existing_item_ids(existing_pallets: list[Any]) -> set[str]:
    item_ids: set[str] = set()
    for pallet_index, raw_pallet in enumerate(existing_pallets):
        pallet = _require_map(raw_pallet, f"existingPallets[{pallet_index}]")
        placements = _require_list(
            pallet.get("placements") or [],
            f"existingPallets[{pallet_index}].placements",
        )
        for placement_index, raw_placement in enumerate(placements):
            placement = _require_map(
                raw_placement,
                f"existingPallets[{pallet_index}].placements[{placement_index}]",
            )
            item_id = _require_str(
                placement.get("itemId"),
                f"existingPallets[{pallet_index}].placements[{placement_index}].itemId",
            )
            if item_id in item_ids:
                raise ProtocolError(
                    2001,
                    f"duplicate itemId in existingPallets: {item_id}",
                    details=[{"code": 2001, "path": "payload.existingPallets", "message": f"duplicate itemId {item_id}"}],
                )
            item_ids.add(item_id)
    return item_ids


def _normalize_existing_placements(existing_pallet: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    placements = _require_list(
        existing_pallet.get("placements") or [],
        f"{existing_pallet.get('palletId', 'existingPallet')}.placements",
    )
    for index, raw_placement in enumerate(placements):
        placement = _require_map(raw_placement, f"existing placement {index}")
        dims = _require_map(
            placement.get("itemDimensions"),
            f"existing placement {index}.itemDimensions",
        )
        pose = _require_map(
            placement.get("pose"),
            f"existing placement {index}.pose",
        )
        normalized.append(
            {
                "seqNo": int(placement.get("seqNo") or len(normalized) + 1),
                "itemId": _require_str(placement.get("itemId"), f"existing placement {index}.itemId"),
                "skuCode": _require_str(placement.get("skuCode"), f"existing placement {index}.skuCode"),
                "itemDimensions": {
                    "length": round(_require_positive_number(dims.get("length"), f"existing placement {index}.itemDimensions.length"), 3),
                    "width": round(_require_positive_number(dims.get("width"), f"existing placement {index}.itemDimensions.width"), 3),
                    "height": round(_require_positive_number(dims.get("height"), f"existing placement {index}.itemDimensions.height"), 3),
                },
                "pose": {
                    "x": round(_to_float(pose.get("x")), 3),
                    "y": round(_to_float(pose.get("y")), 3),
                    "z": round(_to_float(pose.get("z")), 3),
                    "rx": round(_to_float(pose.get("rx")), 3),
                    "ry": round(_to_float(pose.get("ry")), 3),
                    "rz": round(_to_float(pose.get("rz")), 3),
                },
            }
        )
    return normalized


def _normalize_existing_pallet(
    raw_pallet: Any,
    index: int,
) -> ExistingPalletState:
    pallet = _require_map(raw_pallet, f"existingPallets[{index}]")
    pallet_id = _require_str(
        pallet.get("palletId"),
        f"existingPallets[{index}].palletId",
    )
    placements = _normalize_existing_placements(pallet)
    raw_load = pallet.get("loadSummary") or {}
    load_summary = _require_map(raw_load, f"existingPallets[{index}].loadSummary") if raw_load else {}
    total_items = int(load_summary.get("totalItems") or len(placements))
    total_weight = round(_to_float(load_summary.get("totalWeight")), 3)
    final_height = round(
        _to_float(
            load_summary.get("finalHeight"),
            max((_placement_top_z(p) for p in placements), default=0.0),
        ),
        3,
    )
    is_locked = bool(placements) or total_items > 0 or final_height > 0.0
    return ExistingPalletState(
        pallet_id=pallet_id,
        placements=placements,
        load_summary={
            "totalItems": total_items,
            "totalWeight": total_weight,
            "finalHeight": final_height,
        },
        is_locked=is_locked,
    )


def _planned_placement_to_protocol(
    seq_no: int,
    placement: Any,
    original_dims_map: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    size_x_mm = float(placement.size_x) * 1000.0
    size_y_mm = float(placement.size_y) * 1000.0
    size_z_mm = float(placement.size_z) * 1000.0
    x_mm = float(placement.x) * 1000.0
    y_mm = float(placement.y) * 1000.0
    z_mm = float(placement.z) * 1000.0

    original_dims = None
    if original_dims_map is not None:
        original_dims = original_dims_map.get(str(placement.instance_id))
    if original_dims is None:
        out_length_mm, out_width_mm, out_height_mm = size_x_mm, size_y_mm, size_z_mm
    else:
        out_length_mm, out_width_mm, out_height_mm = original_dims

    return {
        "seqNo": seq_no,
        "itemId": placement.instance_id,
        "skuCode": placement.box_type_id,
        "itemDimensions": {
            "length": round(out_length_mm, 3),
            "width": round(out_width_mm, 3),
            "height": round(out_height_mm, 3),
        },
        "pose": {
            "x": round(x_mm, 3),
            "y": round(y_mm, 3),
            "z": round(z_mm, 3),
            "rx": 0.0,
            "ry": 0.0,
            "rz": round(math.degrees(float(placement.yaw)), 3),
        },
    }


def _placement_top_z(placement: dict[str, Any]) -> float:
    dims = placement.get("itemDimensions") or {}
    pose = placement.get("pose") or {}
    if not isinstance(dims, dict) or not isinstance(pose, dict):
        return 0.0
    return _to_float(pose.get("z")) + _to_float(dims.get("height")) / 2.0


def _placement_weight(placement: dict[str, Any], weight_map: dict[str, float]) -> float:
    item_id = placement.get("itemId")
    if isinstance(item_id, str) and item_id in weight_map:
        return weight_map[item_id]
    return 0.0


def _generate_new_pallet_id(used_ids: set[str]) -> str:
    index = 1
    while f"P{index}" in used_ids:
        index += 1
    pallet_id = f"P{index}"
    used_ids.add(pallet_id)
    return pallet_id


def _pallet_mode_to_target_count(
    payload: dict[str, Any],
    total_new_items: int,
) -> int:
    constraints = payload["scene"]["constraints"]
    mode = constraints["palletPolicy"]["mode"]
    if mode == "SINGLE_PALLET":
        return 1
    if mode == "FIXED_N_PALLETS":
        return int(constraints["palletPolicy"]["palletCount"])
    if mode == "UNTIL_ALL_ITEMS_PLACED":
        return max(1, total_new_items)
    raise ProtocolError(
        2001,
        f"scene constraint conflict: unsupported palletPolicy.mode={mode}",
        details=[{"code": 2001, "path": "payload.scene.constraints.palletPolicy.mode", "message": f"unsupported mode {mode}"}],
    )


def _select_candidate_boxes(
    boxes: list[AlgoBox],
    weight_capacity: float,
) -> tuple[list[AlgoBox], list[AlgoBox]]:
    if weight_capacity <= 0.0:
        return [], list(boxes)

    selected: list[AlgoBox] = []
    deferred: list[AlgoBox] = []
    running_weight = 0.0

    for box in boxes:
        if box.weight > weight_capacity + 1e-9:
            deferred.append(box)
            continue
        if running_weight + box.weight <= weight_capacity + 1e-9 or not selected:
            selected.append(box)
            running_weight += box.weight
        else:
            deferred.append(box)

    return selected, deferred


def _solve_one_pallet(
    boxes: list[AlgoBox],
    pallet_spec: PalletSpec,
    time_limit_seconds: float,
    algorithm: str,
    pop_size: int,
) -> SolveOutcome:
    task_boxes: list[TaskBox] = []
    for box in boxes:
        length_m = box.length / 1000.0
        width_m = box.width / 1000.0
        allowed_yaws = box.allowed_yaws

        # The current planners only model "rotation allowed" vs "not allowed".
        # When the protocol forces 90deg yaw, pre-rotate the dimensions and pin yaw to 0.
        if allowed_yaws == (math.pi / 2,):
            length_m, width_m = width_m, length_m
            allowed_yaws = (0.0,)

        task_boxes.append(
            TaskBox(
                instance_id=box.item_id,
                box_type_id=box.sku_code,
                length=length_m,
                width=width_m,
                height=box.height / 1000.0,
                frequency_group=box.sku_code,
                allowed_yaws=allowed_yaws,
            )
        )

    if algorithm == "3d":
        return SolveOutcome(
            plan_result=plan_multitype_palletizing_3d(
                task_boxes,
                pallet_spec,
                time_limit_seconds=max(time_limit_seconds, 5.0),
                num_workers=8,
            ),
            resolved_algorithm="3d",
        )

    solver_warning: str | None = None
    solver_fn = plan_multitype_palletizing_ga3d
    if algorithm == "ga3d-seam":
        solver_fn = plan_multitype_palletizing_ga3d_seam
        solver_warning = "algorithm=ga3d-seam is mapped to the maintained ga3d implementation."

    try:
        return SolveOutcome(
            plan_result=solver_fn(
                task_boxes,
                pallet_spec,
                time_limit_seconds=time_limit_seconds,
                pop_size=pop_size,
            ),
            resolved_algorithm=algorithm,
            warning=solver_warning,
        )
    except Exception as exc:  # pragma: no cover
        fallback = plan_multitype_palletizing_3d(
            task_boxes,
            pallet_spec,
            time_limit_seconds=max(time_limit_seconds, 5.0),
            num_workers=8,
        )
        fallback_warning = (
            solver_warning + " " if solver_warning else ""
        ) + f"{algorithm} failed and the service fell back to 3d: {exc}"
        return SolveOutcome(
            plan_result=fallback,
            resolved_algorithm="3d",
            warning=fallback_warning.strip(),
        )


def _build_pallet_payload(
    pallet_id: str,
    placements: list[dict[str, Any]],
    weight_map: dict[str, float],
    base_weight: float = 0.0,
    total_items_override: int | None = None,
    final_height_override: float | None = None,
) -> dict[str, Any]:
    resequenced: list[dict[str, Any]] = []
    for seq_no, placement in enumerate(placements, start=1):
        updated = dict(placement)
        updated["seqNo"] = seq_no
        resequenced.append(updated)

    total_weight = base_weight + sum(
        _placement_weight(placement, weight_map)
        for placement in resequenced
    )
    final_height = max(
        (_placement_top_z(placement) for placement in resequenced),
        default=final_height_override or 0.0,
    )
    total_items = (
        total_items_override
        if total_items_override is not None
        else len(resequenced)
    )
    return {
        "palletId": pallet_id,
        "loadSummary": {
            "totalItems": total_items,
            "totalWeight": round(total_weight, 3),
            "finalHeight": round(final_height, 3),
        },
        "placements": resequenced,
    }


def handle_algo_request(body: dict[str, Any]) -> dict[str, Any]:
    envelope = _require_map(body, "root")
    trace_id = _require_str(envelope.get("traceId"), "traceId")
    request_protocol_version = _require_str(
        envelope.get("protocolVersion"),
        "protocolVersion",
    )
    message_type = _require_str(envelope.get("messageType"), "messageType")
    if message_type != "AlgoInput":
        raise ProtocolError(
            1002,
            "field type error: messageType should be AlgoInput",
            details=[{"code": 1002, "path": "messageType", "message": "expected AlgoInput"}],
        )

    correlation_id = envelope.get("correlationId")
    if correlation_id is not None and not isinstance(correlation_id, str):
        raise ProtocolError(
            1002,
            "field type error: correlationId should be string",
            details=[{"code": 1002, "path": "correlationId", "message": "expected string"}],
        )

    payload = _require_map(envelope.get("payload"), "payload")
    task_id = _require_str(payload.get("taskId"), "payload.taskId")
    _require_str(payload.get("orderId"), "payload.orderId")
    scene = _require_map(payload.get("scene"), "payload.scene")

    robots = scene.get("robots")
    if robots is not None:
        _require_list(robots, "payload.scene.robots")

    constraints = _require_map(
        scene.get("constraints"),
        "payload.scene.constraints",
    )
    _validate_constraints(constraints)

    warnings = _collect_constraint_warnings(payload, constraints)
    if request_protocol_version != PROTOCOL_VERSION:
        warnings.append(
            f"request protocolVersion={request_protocol_version} was accepted for compatibility; response uses {PROTOCOL_VERSION}."
        )

    pallet_obj = _require_map(scene.get("pallet"), "payload.scene.pallet")
    pallet_length, pallet_width, _ = _get_dims3d(
        pallet_obj.get("dimensions"),
        "payload.scene.pallet.dimensions",
    )
    max_load_height = _require_positive_number(
        pallet_obj.get("maxLoadHeight"),
        "payload.scene.pallet.maxLoadHeight",
    )
    max_load_weight = _require_positive_number(
        pallet_obj.get("maxLoadWeight"),
        "payload.scene.pallet.maxLoadWeight",
    )

    material_items = _require_list(payload.get("materialItems"), "payload.materialItems")
    expanded_boxes, weight_map, original_dims_map = _expand_material_items(material_items, warnings)

    overweight_items = [
        box.item_id
        for box in expanded_boxes
        if box.weight > max_load_weight + 1e-9
    ]
    if overweight_items:
        raise ProtocolError(
            2001,
            f"scene constraint conflict: item exceeds pallet maxLoadWeight: {overweight_items[0]}",
            details=[{"code": 2001, "path": "payload.scene.pallet.maxLoadWeight", "message": f"item {overweight_items[0]} exceeds pallet capacity"}],
        )

    existing_pallets = payload.get("existingPallets") or []
    if not isinstance(existing_pallets, list):
        raise ProtocolError(
            1002,
            "field type error: payload.existingPallets should be array",
            details=[{"code": 1002, "path": "payload.existingPallets", "message": "expected array"}],
        )

    existing_item_ids = _extract_existing_item_ids(existing_pallets)
    duplicate_ids = sorted(existing_item_ids & {box.item_id for box in expanded_boxes})
    if duplicate_ids:
        raise ProtocolError(
            2001,
            f"scene constraint conflict: existingPallets contains itemIds also present in materialItems: {duplicate_ids[0]}",
            details=[{"code": 2001, "path": "payload.existingPallets", "message": f"duplicate itemId {duplicate_ids[0]}"}],
        )

    all_skus = {box.sku_code for box in expanded_boxes}
    for raw_existing in existing_pallets:
        if not isinstance(raw_existing, dict):
            continue
        for raw_placement in raw_existing.get("placements") or []:
            if isinstance(raw_placement, dict) and isinstance(raw_placement.get("skuCode"), str):
                all_skus.add(raw_placement["skuCode"])
    if constraints.get("mixedSkuAllowed") is False and len(all_skus) > 1:
        raise ProtocolError(
            2001,
            "scene constraint conflict: mixedSkuAllowed=false but request contains multiple skuCode values",
            details=[{"code": 2001, "path": "payload.scene.constraints.mixedSkuAllowed", "message": "multiple SKU values are present"}],
        )

    time_limit_seconds = 8.0
    algorithm = "ga3d"
    pop_size = 1024
    extra = payload.get("extra")
    if isinstance(extra, dict):
        solver_extra = extra.get("solver")
        if isinstance(solver_extra, dict):
            raw_limit = solver_extra.get("timeLimitSeconds")
            if isinstance(raw_limit, (int, float)) and not isinstance(raw_limit, bool) and raw_limit > 0:
                time_limit_seconds = float(raw_limit)

            raw_algorithm = solver_extra.get("algorithm")
            if isinstance(raw_algorithm, str) and raw_algorithm.strip():
                algorithm = raw_algorithm.strip().lower()

            raw_pop_size = solver_extra.get("popSize")
            if isinstance(raw_pop_size, int) and raw_pop_size > 0:
                pop_size = raw_pop_size

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ProtocolError(
            1002,
            "invalid value: payload.extra.solver.algorithm should be one of ga3d, ga3d-seam, 3d",
            details=[{"code": 1002, "path": "payload.extra.solver.algorithm", "message": f"unsupported algorithm {algorithm}"}],
        )

    normalized_existing = [
        _normalize_existing_pallet(raw_pallet, index)
        for index, raw_pallet in enumerate(existing_pallets)
    ]
    ordered_pallet_ids = [entry.pallet_id for entry in normalized_existing]
    existing_by_id = {entry.pallet_id: entry for entry in normalized_existing}
    used_pallet_ids = set(ordered_pallet_ids)

    target_pallet_count = _pallet_mode_to_target_count(
        payload,
        total_new_items=len(expanded_boxes),
    )
    if (
        constraints["palletPolicy"]["mode"] != "UNTIL_ALL_ITEMS_PLACED"
        and len(ordered_pallet_ids) > target_pallet_count
    ):
        raise ProtocolError(
            2001,
            "scene constraint conflict: existingPallets count exceeds palletPolicy target",
            details=[{"code": 2001, "path": "payload.existingPallets", "message": "too many existing pallets for current palletPolicy"}],
        )

    if any(entry.is_locked for entry in normalized_existing) and expanded_boxes:
        warnings.append(
            "occupied existingPallets are preserved as locked pallets; incremental in-place packing is not modeled yet."
        )

    if constraints["palletPolicy"]["mode"] != "UNTIL_ALL_ITEMS_PLACED":
        while len(ordered_pallet_ids) < target_pallet_count:
            ordered_pallet_ids.append(_generate_new_pallet_id(used_pallet_ids))

    pallets_out: list[dict[str, Any]] = []
    for entry in normalized_existing:
        if not entry.is_locked:
            continue
        pallets_out.append(
            _build_pallet_payload(
                pallet_id=entry.pallet_id,
                placements=entry.placements,
                weight_map=weight_map,
                base_weight=entry.load_summary["totalWeight"],
                total_items_override=entry.load_summary["totalItems"],
                final_height_override=entry.load_summary["finalHeight"],
            )
        )
        pallets_out[-1]["loadSummary"]["totalWeight"] = entry.load_summary["totalWeight"]
        pallets_out[-1]["loadSummary"]["finalHeight"] = entry.load_summary["finalHeight"]

    remaining_boxes = list(expanded_boxes)
    packed_new_count = 0
    algorithms_used: list[str] = []

    while remaining_boxes:
        reusable_existing = [
            pallet_id
            for pallet_id in ordered_pallet_ids
            if not existing_by_id.get(
                pallet_id,
                ExistingPalletState(
                    pallet_id=pallet_id,
                    placements=[],
                    load_summary={"totalItems": 0, "totalWeight": 0.0, "finalHeight": 0.0},
                    is_locked=False,
                ),
            ).is_locked
            and all(pallet["palletId"] != pallet_id for pallet in pallets_out)
        ]
        if reusable_existing:
            pallet_id = reusable_existing[0]
        elif constraints["palletPolicy"]["mode"] == "UNTIL_ALL_ITEMS_PLACED":
            pallet_id = _generate_new_pallet_id(used_pallet_ids)
            ordered_pallet_ids.append(pallet_id)
        else:
            break

        candidate_boxes, deferred_boxes = _select_candidate_boxes(
            remaining_boxes,
            max_load_weight,
        )
        if not candidate_boxes:
            break

        pallet_spec = PalletSpec(
            length=pallet_length / 1000.0,
            width=pallet_width / 1000.0,
            max_height=max_load_height / 1000.0,
        )
        outcome = _solve_one_pallet(
            boxes=candidate_boxes,
            pallet_spec=pallet_spec,
            time_limit_seconds=time_limit_seconds,
            algorithm=algorithm,
            pop_size=pop_size,
        )
        algorithms_used.append(outcome.resolved_algorithm)
        if outcome.warning:
            warnings.append(outcome.warning)

        planned = outcome.plan_result.placements
        if not planned:
            break

        protocol_placements = [
            _planned_placement_to_protocol(
                seq_no=offset,
                placement=placement,
                original_dims_map=original_dims_map,
            )
            for offset, placement in enumerate(planned, start=1)
        ]
        pallets_out.append(
            _build_pallet_payload(
                pallet_id=pallet_id,
                placements=protocol_placements,
                weight_map=weight_map,
            )
        )

        packed_ids = {placement.instance_id for placement in planned}
        packed_new_count += len(packed_ids)
        remaining_boxes = [
            box
            for box in deferred_boxes + candidate_boxes
            if box.item_id not in packed_ids
        ]

    total_new_items = len(expanded_boxes)
    if packed_new_count == total_new_items:
        plan_status = "SUCCESS"
        fail_reason = None
    elif packed_new_count > 0:
        plan_status = "PARTIAL"
        fail_reason = f"only packed {packed_new_count}/{total_new_items} incoming items"
    else:
        raise ProtocolError(
            3001,
            "no feasible placement found for incoming items",
            http_status=HTTPStatus.UNPROCESSABLE_ENTITY,
            details=[{"code": 3001, "path": "payload.materialItems", "message": "incoming items could not be placed"}],
        )

    diagnostics_messages: list[str] = []
    if algorithms_used:
        diagnostics_messages.append(
            f"algorithms used: {', '.join(algorithms_used)}"
        )
    diagnostics_messages.extend(warnings)

    plan_payload: dict[str, Any] = {
        "taskId": task_id,
        "planId": f"PLAN-{task_id}",
        "planStatus": plan_status,
        "pallets": pallets_out,
        "diagnostics": {
            "placementCount": sum(len(pallet["placements"]) for pallet in pallets_out),
            "palletCount": len(pallets_out),
            "messages": diagnostics_messages,
            "extra": {
                "solverAlgorithmRequested": algorithm,
                "solverAlgorithmsUsed": algorithms_used,
                "requestedIncomingItemCount": total_new_items,
                "packedIncomingItemCount": packed_new_count,
                "existingPalletCount": len(normalized_existing),
            },
        },
        "failReason": fail_reason,
    }

    response = {
        "protocolVersion": PROTOCOL_VERSION,
        "messageType": "PalletPlan",
        "sourceSystem": SOURCE_SYSTEM,
        "schemaVersion": PALLET_PLAN_SCHEMA_VERSION,
        "traceId": trace_id,
        "timestamp": _now_iso(),
        "payload": plan_payload,
    }
    if isinstance(correlation_id, str) and correlation_id:
        response["correlationId"] = correlation_id
    return response


class ProtocolHandler(BaseHTTPRequestHandler):
    server_version = "PalletizingProtocolHTTP/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body_text = json.dumps(payload, ensure_ascii=False)
        body = body_text.encode("utf-8")
        print(
            f"[{_now_iso()}] outgoing {self.command} {self.path}, status={status}, bytes={len(body)}",
            flush=True,
        )
        print(
            json.dumps(payload, ensure_ascii=False, indent=2),
            flush=True,
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_request_body(
        self,
        raw_body: bytes,
        parsed_body: dict[str, Any] | None = None,
    ) -> None:
        print(
            f"[{_now_iso()}] incoming {self.command} {self.path}, bytes={len(raw_body)}",
            flush=True,
        )
        if parsed_body is not None:
            print(
                json.dumps(parsed_body, ensure_ascii=False, indent=2),
                flush=True,
            )
            return
        print(raw_body.decode("utf-8", errors="replace"), flush=True)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {"status": "ok", "timestamp": _now_iso()},
            )
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"code": 1001, "message": f"unknown path: {self.path}", "traceId": ""},
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != ALGO_ENDPOINT:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"code": 1001, "message": f"unknown path: {self.path}", "traceId": ""},
            )
            return

        trace_id = ""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            if not raw_body:
                raise ProtocolError(1001, "missing request body")

            raw_text = raw_body.decode("utf-8", errors="replace")
            try:
                body = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                self._log_request_body(raw_body)
                raise ProtocolError(1002, f"invalid json: {exc.msg}") from exc

            self._log_request_body(raw_body, parsed_body=body if isinstance(body, dict) else None)
            if isinstance(body, dict):
                raw_trace = body.get("traceId")
                if isinstance(raw_trace, str):
                    trace_id = raw_trace

            result = handle_algo_request(body)
            self._write_json(HTTPStatus.OK, result)
        except ProtocolError as exc:
            error_payload = {
                "code": exc.code,
                "message": exc.message,
                "traceId": trace_id,
            }
            if exc.details:
                error_payload["details"] = exc.details
            self._write_json(exc.http_status, error_payload)
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "code": 3001,
                    "message": f"internal error: {exc}",
                    "traceId": trace_id,
                },
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mixed protocol HTTP server for palletizing algorithm",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ProtocolHandler)
    print(f"HTTP server listening on http://{args.host}:{args.port}")
    print(f"POST {ALGO_ENDPOINT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
