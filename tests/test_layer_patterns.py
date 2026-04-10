import unittest

from palletizing import LayerPlacement, PalletSpec, TaskBox, build_layer_blocks, plan_best_layer_pattern, plan_layer_pattern_for_height
from palletizing.geometry import bottom_alignment_corners, rect_corners
from palletizing.layer_patterns import _gap_fill_score


class LayerPatternTests(unittest.TestCase):
    def test_build_layer_blocks_includes_matching_singles_and_composites(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_003", "B", 0.3, 0.2, 0.2, "medium", (0.0, 1.5707963267948966)),
        ]

        blocks = build_layer_blocks(boxes, target_height=0.2, max_stack_size=2)

        self.assertTrue(any(not block.is_composite for block in blocks))
        self.assertTrue(any(block.is_composite for block in blocks))

    def test_plan_layer_pattern_packs_disjoint_blocks(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_003", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_004", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
        ]
        pallet = PalletSpec(1.2, 1.0, 1.5)

        result = plan_layer_pattern_for_height(boxes, pallet, target_height=0.2, max_stack_size=1)

        self.assertEqual(result.used_box_count, 4)
        self.assertAlmostEqual(result.utilization_2d, 1.0, places=6)
        self.assertEqual(len(result.used_box_instance_ids), len(set(result.used_box_instance_ids)))

    def test_plan_layer_pattern_respects_support_placements(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
        ]
        pallet = PalletSpec(1.2, 1.0, 1.5)
        support = [
            LayerPlacement("support", ("box_s",), ("S",), 0.3, 0.25, 0.0, 0.6, 0.5, 0.2, 1, False),
        ]

        result = plan_layer_pattern_for_height(
            boxes,
            pallet,
            target_height=0.2,
            max_stack_size=2,
            support_placements=support,
            min_support_ratio=0.9,
        )

        self.assertEqual(result.packed_block_count, 1)

    def test_bottom_alignment_corners_generate_three_choices_per_anchor(self) -> None:
        corners = rect_corners(0.6, 0.5)
        aligned = bottom_alignment_corners(corners, 0)

        self.assertEqual(len(aligned), 3)
        self.assertNotIn(corners[0], aligned)

    def test_best_layer_pattern_prefers_full_pallet_fill(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_003", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_004", "A", 0.6, 0.5, 0.2, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_005", "B", 0.6, 0.5, 0.3, "medium", (0.0, 1.5707963267948966)),
            TaskBox("box_006", "B", 0.6, 0.5, 0.3, "medium", (0.0, 1.5707963267948966)),
        ]
        pallet = PalletSpec(1.2, 1.0, 1.5)

        result = plan_best_layer_pattern(boxes, pallet, max_stack_size=1)

        self.assertAlmostEqual(result.utilization_2d, 1.0, places=6)
        self.assertEqual(result.target_height, 0.2)

    def test_gap_fill_score_prefers_inside_bbox_fill(self) -> None:
        placements = (
            LayerPlacement("a", ("a",), ("A",), 0.3, 0.25, 0.0, 0.6, 0.5, 0.2, 1, False),
            LayerPlacement("b", ("b",), ("A",), 0.9, 0.25, 0.0, 0.6, 0.5, 0.2, 1, False),
            LayerPlacement("c", ("c",), ("A",), 0.3, 0.75, 0.0, 0.6, 0.5, 0.2, 1, False),
        )
        inside = LayerPlacement("inside", ("d",), ("A",), 0.9, 0.75, 0.0, 0.6, 0.5, 0.2, 1, False)
        outside = LayerPlacement("outside", ("e",), ("A",), 1.5, 0.75, 0.0, 0.6, 0.5, 0.2, 1, False)

        self.assertGreater(_gap_fill_score(inside, placements), _gap_fill_score(outside, placements))



if __name__ == "__main__":
    unittest.main()
