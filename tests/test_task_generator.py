import unittest

from palletizing import BOX_TYPE_CATALOG, generate_multitype_task


class TaskGeneratorTests(unittest.TestCase):
    def test_catalog_contains_all_candidate_box_types(self) -> None:
        self.assertEqual(len(BOX_TYPE_CATALOG), 60)
        self.assertEqual(BOX_TYPE_CATALOG[0].box_type_id, "H00")
        self.assertEqual(BOX_TYPE_CATALOG[-1].box_type_id, "L19")

    def test_generate_multitype_task_returns_requested_count(self) -> None:
        task = generate_multitype_task(count=70, seed=7)

        self.assertEqual(len(task.boxes), 70)
        self.assertEqual(task.summary["count"], 70)
        self.assertGreaterEqual(task.summary["unique_box_type_count"], 1)

    def test_generation_is_reproducible_with_seed(self) -> None:
        task_a = generate_multitype_task(count=10, seed=123)
        task_b = generate_multitype_task(count=10, seed=123)

        ids_a = [box.box_type_id for box in task_a.boxes]
        ids_b = [box.box_type_id for box in task_b.boxes]
        self.assertEqual(ids_a, ids_b)


if __name__ == "__main__":
    unittest.main()
