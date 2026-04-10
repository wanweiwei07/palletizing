import unittest

from palletizing import BoxSpec, PalletSpec, plan_palletizing


class PlannerTests(unittest.TestCase):
    def test_plans_all_70_boxes_when_capacity_is_sufficient(self) -> None:
        pallet = PalletSpec(length=1200, width=1000, max_height=1500)
        box = BoxSpec(length=300, width=200, height=250, count=70)

        result = plan_palletizing(pallet, box)

        self.assertEqual(result.packed_count, 70)
        self.assertEqual(len(result.placements), 70)
        self.assertGreaterEqual(result.layer_count, 1)
        self.assertLessEqual(result.placements[-1].z + result.placements[-1].size_z / 2, pallet.max_height)

    def test_returns_partial_plan_when_not_all_boxes_fit(self) -> None:
        pallet = PalletSpec(length=500, width=400, max_height=300)
        box = BoxSpec(length=260, width=210, height=200, count=70)

        result = plan_palletizing(pallet, box)

        self.assertLess(result.packed_count, 70)
        self.assertEqual(result.packed_count, len(result.placements))


if __name__ == "__main__":
    unittest.main()
