import unittest

from palletizing import PalletSpec, generate_multitype_task, plan_multitype_palletizing


class MultiTypePlannerTests(unittest.TestCase):
    def test_multitype_planner_returns_feasible_positions(self) -> None:
        task = generate_multitype_task(count=20, seed=7)
        pallet = PalletSpec(length=1.2, width=1.0, max_height=1.5)

        result = plan_multitype_palletizing(task.boxes, pallet)

        self.assertGreater(result.packed_count, 0)
        self.assertLessEqual(result.packed_count, 20)
        for placement in result.placements:
            self.assertLessEqual(placement.x - placement.size_x / 2, placement.x)
            self.assertGreaterEqual(placement.x - placement.size_x / 2, 0.0)
            self.assertLessEqual(placement.x + placement.size_x / 2, pallet.length + 1e-9)
            self.assertGreaterEqual(placement.y - placement.size_y / 2, 0.0)
            self.assertLessEqual(placement.y + placement.size_y / 2, pallet.width + 1e-9)
            self.assertLessEqual(placement.z + placement.size_z / 2, pallet.max_height + 1e-9)

    def test_multitype_planner_outputs_non_decreasing_layer_sequence(self) -> None:
        task = generate_multitype_task(count=12, seed=7)
        pallet = PalletSpec(length=1.2, width=1.0, max_height=1.5)

        result = plan_multitype_palletizing(task.boxes, pallet)

        z_values = [placement.z for placement in result.placements]
        self.assertEqual(z_values, sorted(z_values))


if __name__ == "__main__":
    unittest.main()
