"""Regressions from the independent review of the QC-feedback branch (2026-10-22).

Each test reproduces one finding of the review (its experiment in the review's scratch folder) and fails on the code
before its fix. Every provider is a local fake.
"""

from __future__ import annotations

import unittest

from app.idm.quiz_text import QUESTION_LAYOUT_CODE, layout_question, question_report


def words(text: str) -> list[str]:
    """The words of ``text`` without list markers (the layout must keep every word)."""

    return [word for word in text.split() if word not in {"-", "—", "–"}]  # noqa: RUF001 - dashes


class InlineListTests(unittest.TestCase):
    """M1: the inline-list split broke ranges ("10 - 20 triệu"), times ("8:00 - 12:00") and pairs ("Hà Nội - Hải
    Phòng"); a list is split only after an intro ending in ":" and at items that start alike ("Mức 1:", "Mức 2:")."""

    def test_ranges_stay_inside_their_item(self) -> None:
        stem = ("Các mức bồi thường như sau: - Mức 1: 10 - 20 triệu đồng - Mức 2: 20 - 50 triệu đồng. Khách hàng X "
                "thiệt hại 30 triệu. Áp dụng mức nào?")
        laid = layout_question(stem)
        self.assertEqual(laid.split("\n"), [
            "Các mức bồi thường như sau:", "- Mức 1: 10 - 20 triệu đồng", "- Mức 2: 20 - 50 triệu đồng.",
            "Khách hàng X thiệt hại 30 triệu.", "Áp dụng mức nào?"])
        self.assertEqual(words(laid), words(stem))
        self.assertNotIn(QUESTION_LAYOUT_CODE, question_report({"question": laid}))
        en_dash = "Các mức như sau: - Mức 1: từ 10 – 20 triệu - Mức 2: từ 20 – 50 triệu. Chọn mức nào?"  # noqa: RUF001
        self.assertEqual(layout_question(en_dash).split("\n")[1:3],
                         ["- Mức 1: từ 10 – 20 triệu", "- Mức 2: từ 20 – 50 triệu."])  # noqa: RUF001

    def test_times_and_place_pairs_stay_whole(self) -> None:
        shifts = "Các ca như sau: - Ca sáng: 8:00 - 12:00 - Ca chiều: 13:00 - 17:00. Bạn chọn ca nào?"
        self.assertEqual(layout_question(shifts).split("\n"), [
            "Các ca như sau:", "- Ca sáng: 8:00 - 12:00", "- Ca chiều: 13:00 - 17:00.", "Bạn chọn ca nào?"])
        routes = "Các tuyến như sau: - Tuyến 1: Hà Nội - Hải Phòng - Tuyến 2: Hà Nội - Nam Định. Chọn tuyến nào?"
        laid = layout_question(routes)
        self.assertEqual(laid.split("\n"), [
            "Các tuyến như sau:", "- Tuyến 1: Hà Nội - Hải Phòng", "- Tuyến 2: Hà Nội - Nam Định.",
            "Chọn tuyến nào?"])
        self.assertEqual(question_report({"question": laid}), {})

    def test_an_unsure_list_is_left_as_written_and_reported(self) -> None:
        for stem in ("Thông tin: - Khách B2B - Doanh nghiệp lớn - Khách B2C - Cá nhân. Bạn chọn kênh nào?",
                     "Bối cảnh: - Tỷ lệ giảm - 5% so với quý trước - Chi phí tăng. Bạn nên làm gì?"):
            laid = layout_question(stem)
            self.assertEqual(laid.split("\n")[0], stem.rsplit(". ", 1)[0] + ".")
            self.assertEqual(words(laid), words(stem))
            self.assertIn(QUESTION_LAYOUT_CODE, question_report({"question": laid}))
        # Items that start alike are cut; what follows the last label without a label of its own is reported.
        mixed = layout_question("Theo bảng: - Nhóm X: doanh thu cao - Nhóm Y: doanh thu thấp - Ghi chú - Xem phụ lục "
                                "2. Nhóm nào được ưu tiên?")
        self.assertEqual(mixed.splitlines()[:3], ["Theo bảng:", "- Nhóm X: doanh thu cao",
                                                  "- Nhóm Y: doanh thu thấp - Ghi chú - Xem phụ lục 2."])
        self.assertIn(QUESTION_LAYOUT_CODE, question_report({"question": mixed}))


if __name__ == "__main__":
    unittest.main()
