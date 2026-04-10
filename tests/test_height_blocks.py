import unittest

from palletizing import TaskBox, analyze_box_heights


class HeightAnalysisTests(unittest.TestCase):
    def test_groups_boxes_by_exact_height(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_003", "B", 0.2, 0.2, 0.2, "medium", (0.0, 1.5707963267948966)),
        ]

        analysis = analyze_box_heights(boxes)

        self.assertEqual(len(analysis.height_groups), 2)
        self.assertEqual(analysis.height_groups[0].target_height, 0.1)
        self.assertEqual(analysis.height_groups[0].count, 2)
        self.assertEqual(analysis.height_groups[1].target_height, 0.2)
        self.assertEqual(analysis.height_groups[1].count, 1)

    def test_finds_stacked_combinations_matching_existing_target_height(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_003", "B", 0.2, 0.2, 0.2, "medium", (0.0, 1.5707963267948966)),
        ]

        analysis = analyze_box_heights(boxes)
        candidates = analysis.block_candidates_by_target[0.2]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].box_instance_ids, ("box_001", "box_002"))
        self.assertEqual(candidates[0].component_count, 2)

    def test_rejects_invalid_combination_size(self) -> None:
        boxes = [
            TaskBox("box_001", "A", 0.3, 0.2, 0.1, "high", (0.0, 1.5707963267948966)),
            TaskBox("box_002", "B", 0.2, 0.2, 0.2, "medium", (0.0, 1.5707963267948966)),
        ]

        with self.assertRaises(ValueError):
            analyze_box_heights(boxes, max_combination_size=1)


if __name__ == "__main__":
    unittest.main()
