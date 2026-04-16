from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

from protocol_http_server import (
    ALGO_ENDPOINT,
    ProtocolError,
    ProtocolHandler,
    SolveOutcome,
    ThreadingHTTPServer,
    handle_algo_request,
)


def _build_request(
    *,
    material_items: list[dict] | None = None,
    existing_pallets: list[dict] | None = None,
    pallet_policy: dict | None = None,
    extra: dict | None = None,
) -> dict:
    return {
        "protocolVersion": "1.5.0",
        "messageType": "AlgoInput",
        "traceId": "trace-001",
        "sourceSystem": "wms",
        "correlationId": "corr-001",
        "timestamp": "2026-04-15T20:00:00+08:00",
        "payload": {
            "taskId": "TASK-001",
            "orderId": "ORD-001",
            "materialItems": material_items
            or [
                {
                    "itemId": "ITEM-001",
                    "skuCode": "SKU-001",
                    "qty": 1,
                    "dimensions": {
                        "length": 300,
                        "width": 200,
                        "height": 150,
                    },
                    "weight": 5.0,
                    "orientationRules": {
                        "canRotateX": False,
                        "canRotateY": False,
                        "canRotateZ": True,
                    },
                }
            ],
            "existingPallets": existing_pallets or [],
            "scene": {
                "pallet": {
                    "palletType": "EURO",
                    "dimensions": {
                        "length": 1200,
                        "width": 800,
                        "height": 144,
                    },
                    "maxLoadWeight": 1000,
                    "maxLoadHeight": 1800,
                },
                "constraints": {
                    "materialKnowledge": {
                        "mode": "ALL_KNOWN_REORDERABLE",
                        "reorderAllowed": True,
                    },
                    "palletPolicy": pallet_policy
                    or {
                        "mode": "SINGLE_PALLET",
                    },
                    "bufferPolicy": {
                        "mode": "NO_BUFFER",
                    },
                },
            },
            "extra": extra or {"solver": {"timeLimitSeconds": 0.2}},
        },
    }


class ProtocolHttpServerTests(unittest.TestCase):
    def test_handle_algo_request_returns_pallet_plan_for_ga3d(self) -> None:
        response = handle_algo_request(_build_request())

        self.assertEqual(response["protocolVersion"], "1.5.0")
        self.assertEqual(response["messageType"], "PalletPlan")
        self.assertEqual(response["sourceSystem"], "algo")
        self.assertEqual(response["schemaVersion"], "PalletPlan-1.5.0")
        self.assertEqual(response["traceId"], "trace-001")
        self.assertEqual(response["correlationId"], "corr-001")

        payload = response["payload"]
        self.assertEqual(payload["planStatus"], "SUCCESS")
        self.assertEqual(len(payload["pallets"]), 1)
        self.assertEqual(payload["pallets"][0]["placements"][0]["itemId"], "ITEM-001")
        self.assertIn("diagnostics", payload)
        self.assertGreaterEqual(payload["diagnostics"]["placementCount"], 1)

    def test_existing_locked_pallet_is_preserved_and_empty_existing_pallet_is_reused(self) -> None:
        existing_pallets = [
            {
                "palletId": "P-LOCKED",
                "loadSummary": {
                    "totalItems": 1,
                    "totalWeight": 8.0,
                    "finalHeight": 180.0,
                },
                "placements": [
                    {
                        "seqNo": 1,
                        "itemId": "ITEM-OLD",
                        "skuCode": "SKU-OLD",
                        "itemDimensions": {
                            "length": 300,
                            "width": 200,
                            "height": 180,
                        },
                        "pose": {
                            "x": 150,
                            "y": 100,
                            "z": 90,
                            "rx": 0,
                            "ry": 0,
                            "rz": 0,
                        },
                    }
                ],
            },
            {
                "palletId": "P-EMPTY",
                "loadSummary": {
                    "totalItems": 0,
                    "totalWeight": 0.0,
                    "finalHeight": 0.0,
                },
                "placements": [],
            },
        ]
        fake_placement = SimpleNamespace(
            instance_id="ITEM-NEW",
            box_type_id="SKU-NEW",
            size_x=0.4,
            size_y=0.2,
            size_z=0.15,
            x=0.2,
            y=0.1,
            z=0.075,
            yaw=0.0,
        )
        fake_outcome = SolveOutcome(
            plan_result=SimpleNamespace(placements=[fake_placement]),
            resolved_algorithm="ga3d",
        )

        with patch("protocol_http_server._solve_one_pallet", return_value=fake_outcome) as solve_mock:
            response = handle_algo_request(
                _build_request(
                    material_items=[
                        {
                            "itemId": "ITEM-NEW",
                            "skuCode": "SKU-NEW",
                            "qty": 1,
                            "dimensions": {
                                "length": 400,
                                "width": 200,
                                "height": 150,
                            },
                            "weight": 5.0,
                        }
                    ],
                    existing_pallets=existing_pallets,
                    pallet_policy={"mode": "FIXED_N_PALLETS", "palletCount": 2},
                    extra={"solver": {"algorithm": "ga3d", "timeLimitSeconds": 0.1}},
                )
            )

        solve_mock.assert_called_once()
        pallets = response["payload"]["pallets"]
        self.assertEqual([pallet["palletId"] for pallet in pallets], ["P-LOCKED", "P-EMPTY"])
        self.assertEqual(pallets[0]["placements"][0]["itemId"], "ITEM-OLD")
        self.assertEqual(pallets[1]["placements"][0]["itemId"], "ITEM-NEW")

    def test_invalid_orientation_code_raises_protocol_error(self) -> None:
        request_body = _build_request(
            material_items=[
                {
                    "itemId": "ITEM-001",
                    "skuCode": "SKU-001",
                    "qty": 1,
                    "dimensions": {
                        "length": 300,
                        "width": 200,
                        "height": 150,
                    },
                    "weight": 5.0,
                    "orientationRules": {
                        "allowedOrientations": ["xyz", "xzy"],
                    },
                }
            ]
        )

        with self.assertRaises(ProtocolError) as context:
            handle_algo_request(request_body)

        self.assertEqual(context.exception.code, 1004)

    def test_health_endpoint_is_available(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/health"
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["status"], "ok")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
