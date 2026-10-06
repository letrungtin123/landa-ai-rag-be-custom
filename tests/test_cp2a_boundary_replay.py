from __future__ import annotations

import unittest

from tests.cp2a_boundary_replay_bridge import load_fixture, replay_fixture


class Cp2aBoundaryReplayTests(unittest.TestCase):
    def test_frozen_cases_stop_at_the_expected_first_boundary(self) -> None:
        fixture = load_fixture()
        replay = replay_fixture()
        expected = {case["id"]: case["expected"] for case in fixture["cases"]}
        actual = {result["case_id"]: result for result in replay["results"]}

        self.assertEqual(set(actual), set(expected))
        for case_id, contract in expected.items():
            with self.subTest(case_id=case_id):
                result = actual[case_id]
                self.assertEqual(result["status"], contract["status"])
                self.assertEqual(result["first_failing_boundary"], contract["first_failing_boundary"])
                self.assertEqual(result["classification"], contract["classification"])
                if "code" in contract:
                    self.assertEqual(result["code"], contract["code"])

    def test_reviewer_valid_candidates_survive_schema_binding_and_python_validation(self) -> None:
        accepted = [result for result in replay_fixture()["results"] if result["status"] == "accepted"]
        self.assertEqual([result["case_id"] for result in accepted], [
            "text_valid", "table_valid", "procedure_valid", "quiz_valid",
        ])
        for result in accepted:
            with self.subTest(case_id=result["case_id"]):
                self.assertTrue(result["schema_diagnostics"]["schema_valid"])
                self.assertIsNotNone(result["unit"])
                self.assertEqual(
                    [component["type"] for component in result["unit"]["components"]],
                    [plan["type"] for plan in result["contract"]["component_plan"]],
                )

    def test_image_case_is_not_misreported_as_writer_or_validator_failure(self) -> None:
        result = next(
            item for item in replay_fixture()["results"]
            if item["case_id"] == "image_situation_missing_asset"
        )
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["classification"], "input_missing")
        self.assertIsNone(result["unit"])


if __name__ == "__main__":
    unittest.main()
