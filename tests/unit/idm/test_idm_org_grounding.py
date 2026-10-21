"""The specific details of the organization profile and the source that states them (QC run of 2026-10-21).

``app.idm.org_grounding``: the token rule, the source match and the distilled-profile filter. Pure functions; no
provider, no database.
"""

from __future__ import annotations

import unicodedata
import unittest

from app.idm.org_grounding import SourceText, ground_profile, specific_tokens
from app.idm.org_profile import IdmOrgProfileV1
from tests.support.org_context_support import PROFILE, profile

EMPTY = profile(organization_summary="", products_services=[], learner_groups=[], glossary=[], key_processes=[],
                culture_tone="", design_conventions=[])


class SpecificTokenTests(unittest.TestCase):
    def test_numbers_codes_and_acronyms_only(self) -> None:
        text = ("Tuân thủ QCVN 40:2021/BTNMT, xin IRC/ERC (nếu cần); dự án 3.400 ha nối lưới 110kV/22kV; chứng chỉ "
                "I-REC; giám sát SCADA/OCC tại TP.HCM; giảm 20% chi phí. Công ty Hải Đăng, kV, IoT, Wi-Fi, A.")
        self.assertEqual(specific_tokens(text), ["QCVN", "40:2021", "BTNMT", "IRC", "ERC", "3.400", "110kV", "22kV",
                                                 "I-REC", "SCADA", "OCC", "TP.HCM", "20", "IoT"])

    def test_ordinary_words_are_never_checked(self) -> None:
        self.assertEqual(specific_tokens("Công ty Hải Đăng Logistics vận hành kho lạnh tại miền Trung."), [])
        self.assertEqual(specific_tokens(""), [])


class SourceTextTests(unittest.TestCase):
    def test_a_whole_token_matches_after_nfc_case_folding_and_collapsed_spaces(self) -> None:
        source = SourceText.of([unicodedata.normalize("NFD", "Ghi  vào Mẫu SỔ-01\n\ncủa trạm."),
                                "Nhà đầu tư xin IRC trước khi khởi công."])
        self.assertTrue(source.states("sổ-01"))
        self.assertTrue(source.states("irc"))
        self.assertTrue(source.states("IRC"))
        # Inside a longer word ("commerce", "IRCs") it is not the token.
        self.assertFalse(SourceText.of(["Commerce exercise: IRCs."]).states("ERC"))
        self.assertFalse(SourceText.of(["IRCs"]).states("IRC"))
        self.assertFalse(source.states("ERC"))
        self.assertFalse(source.states(""))

    def test_numbers_also_match_on_their_digits(self) -> None:
        source = SourceText.of(["Dự án rộng 3400 ha, phí 20,000 đồng, nối lưới 110 kV và 22kV; QCVN 40:2021/BTNMT."])
        for token in ("3.400", "3400", "20.000", "20000", "110kV", "22kV", "40:2021", "BTNMT"):
            with self.subTest(token=token):
                self.assertTrue(source.states(token))
        for token in ("13.400", "340", "35kV", "40:2022", "2020"):
            with self.subTest(token=token):
                self.assertFalse(source.states(token))
        self.assertFalse(SourceText.of(["Dự án rộng 13.400 ha."]).states("3.400"))


SOURCE = ["Dự án điện mặt trời Phước Hòa rộng 3400 ha, đấu nối lưới 110kV qua trạm biến áp.",
          "Đường dây 22kV cấp điện cho khu vận hành. Nhà đầu tư xin IRC trước khi khởi công.",
          "Trung tâm vận hành giám sát bằng SCADA, tuân thủ QCVN 40:2021/BTNMT."]


class GroundProfileTests(unittest.TestCase):
    def test_items_and_sentences_with_an_unstated_detail_are_dropped(self) -> None:
        drafted = IdmOrgProfileV1.model_validate(profile(
            organization_summary=("Công ty Phước Hòa vận hành dự án điện mặt trời 3.400 ha. Công ty đạt chứng chỉ "
                                  "I-REC năm 2020.\nTrung tâm vận hành giám sát bằng SCADA."),
            products_services=["Điện mặt trời Phước Hòa: 3.400 ha nối lưới 110kV/22kV."],
            learner_groups=[{"name": "Kỹ sư vận hành", "description": "Giám sát trạm 22kV qua SCADA."},
                            {"name": "Nhóm OCC", "description": "Điều độ vận hành."}],
            glossary=[{"term": "IRC", "meaning": "Giấy chứng nhận đăng ký đầu tư."},
                      {"term": "ERC", "meaning": "Giấy chứng nhận đăng ký doanh nghiệp."},
                      {"term": "Trạm 110kV", "meaning": "Trạm biến áp đấu nối, công suất 50MVA."}],
            key_processes=["Xin IRC trước khi khởi công.", "Xin IRC/ERC cho dự án.", "Bảo trì trạm mỗi 6 tháng.",
                           "Vận hành lưới 110kV/22kV theo QCVN 40:2021/BTNMT."],
            culture_tone="An toàn là trên hết. Báo cáo sự cố trong 2 giờ.",
            design_conventions=["Mỗi bài có một tình huống ở trạm 110kV."]))
        grounded, removed = ground_profile(drafted, SOURCE)
        self.assertEqual(grounded.model_dump(mode="json"), profile(
            organization_summary=("Công ty Phước Hòa vận hành dự án điện mặt trời 3.400 ha.\nTrung tâm vận hành giám "
                                  "sát bằng SCADA."),
            products_services=["Điện mặt trời Phước Hòa: 3.400 ha nối lưới 110kV/22kV."],
            learner_groups=[{"name": "Kỹ sư vận hành", "description": "Giám sát trạm 22kV qua SCADA."}],
            glossary=[{"term": "IRC", "meaning": "Giấy chứng nhận đăng ký đầu tư."}],
            key_processes=["Xin IRC trước khi khởi công.", "Vận hành lưới 110kV/22kV theo QCVN 40:2021/BTNMT."],
            culture_tone="An toàn là trên hết.",
            design_conventions=["Mỗi bài có một tình huống ở trạm 110kV."]))
        self.assertEqual(removed, {"culture_tone": 1, "glossary": 2, "key_processes": 2, "learner_groups": 1,
                                   "organization_summary": 1})

    def test_a_grounded_profile_is_returned_as_it_is(self) -> None:
        drafted = IdmOrgProfileV1.model_validate(PROFILE)
        grounded, removed = ground_profile(drafted, ["Phiếu HD-07 là biên bản bàn giao hàng lạnh."])
        self.assertIs(grounded, drafted)
        self.assertEqual(removed, {})

    def test_a_profile_left_empty_is_still_a_valid_draft(self) -> None:
        drafted = IdmOrgProfileV1.model_validate(profile(
            organization_summary="Công ty có 120 nhân viên.\nĐạt ISO 9001.", products_services=["Gói ABC"],
            learner_groups=[{"name": "Nhóm KCS", "description": ""}], glossary=[{"term": "KPI", "meaning": "Chỉ số."}],
            key_processes=["Quy trình 5S."], culture_tone="Phục vụ 24/7.", design_conventions=["Bài dài 10 phút."]))
        grounded, removed = ground_profile(drafted, ["Tài liệu không nêu con số nào."])
        self.assertEqual(grounded.model_dump(mode="json"), EMPTY)
        self.assertEqual(removed, {"culture_tone": 1, "design_conventions": 1, "glossary": 1, "key_processes": 1,
                                   "learner_groups": 1, "organization_summary": 2, "products_services": 1})


if __name__ == "__main__":
    unittest.main()
