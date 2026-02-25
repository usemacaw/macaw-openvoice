"""Tests for the entity detection module (PII/PHI/PCI regex detector).

Covers DetectedEntity dataclass, regex patterns for each entity type,
category filtering, sorting, and the factory function.
"""

from __future__ import annotations

from macaw.postprocessing.entity_detection import (
    DetectedEntity,
    EntityDetector,
    RegexEntityDetector,
    create_entity_detector,
)
from macaw.postprocessing.entity_detection.patterns import ALL_CATEGORIES, ENTITY_PATTERNS


class TestDetectedEntityDataclass:
    """DetectedEntity is a frozen dataclass with correct fields."""

    def test_fields_are_accessible(self) -> None:
        entity = DetectedEntity(
            text="test@example.com",
            entity_type="email_address",
            category="pii",
            start_char=0,
            end_char=16,
        )
        assert entity.text == "test@example.com"
        assert entity.entity_type == "email_address"
        assert entity.category == "pii"
        assert entity.start_char == 0
        assert entity.end_char == 16

    def test_frozen_prevents_mutation(self) -> None:
        entity = DetectedEntity(text="x", entity_type="t", category="c", start_char=0, end_char=1)
        try:
            entity.text = "y"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestEntityPatterns:
    """Each compiled regex matches expected inputs."""

    def test_all_categories_contains_three(self) -> None:
        assert frozenset({"pii", "phi", "pci"}) == ALL_CATEGORIES

    def test_patterns_tuple_is_not_empty(self) -> None:
        assert len(ENTITY_PATTERNS) > 0

    def test_all_patterns_have_valid_category(self) -> None:
        for ep in ENTITY_PATTERNS:
            assert ep.category in ALL_CATEGORIES, f"{ep.entity_type} has invalid category"


class TestRegexDetectorEmails:
    """Email pattern detection."""

    def test_finds_simple_email(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Contact us at info@example.com today")
        assert len(result) == 1
        assert result[0].entity_type == "email_address"
        assert result[0].text == "info@example.com"
        assert result[0].category == "pii"

    def test_finds_email_with_dots_and_plus(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Email john.doe+tag@sub.example.co.uk please")
        assert len(result) == 1
        assert result[0].text == "john.doe+tag@sub.example.co.uk"

    def test_no_false_positive_on_plain_text(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("No email addresses here at all")
        emails = [e for e in result if e.entity_type == "email_address"]
        assert len(emails) == 0


class TestRegexDetectorPhones:
    """US phone number pattern detection."""

    def test_finds_parenthesized_phone(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Call (555) 123-4567 for info")
        phones = [e for e in result if e.entity_type == "phone_number"]
        assert len(phones) == 1
        assert phones[0].text == "(555) 123-4567"

    def test_finds_dashed_phone(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Number: 555-123-4567")
        phones = [e for e in result if e.entity_type == "phone_number"]
        assert len(phones) == 1
        assert phones[0].text == "555-123-4567"

    def test_finds_phone_with_country_code(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Call +1-555-123-4567")
        phones = [e for e in result if e.entity_type == "phone_number"]
        assert len(phones) == 1
        assert "+1" in phones[0].text


class TestRegexDetectorSSN:
    """SSN pattern detection."""

    def test_finds_ssn(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("SSN is 123-45-6789 on file")
        ssns = [e for e in result if e.entity_type == "ssn"]
        assert len(ssns) == 1
        assert ssns[0].text == "123-45-6789"

    def test_no_false_positive_on_phone(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Phone 555-123-4567")
        ssns = [e for e in result if e.entity_type == "ssn"]
        assert len(ssns) == 0


class TestRegexDetectorMRN:
    """Medical Record Number pattern detection."""

    def test_finds_mrn_with_colon(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Patient MRN: 12345678")
        mrns = [e for e in result if e.entity_type == "medical_record_number"]
        assert len(mrns) == 1
        assert "12345678" in mrns[0].text

    def test_finds_mrn_with_hash(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Record MRN#987654")
        mrns = [e for e in result if e.entity_type == "medical_record_number"]
        assert len(mrns) == 1

    def test_case_insensitive(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("mrn 555555")
        mrns = [e for e in result if e.entity_type == "medical_record_number"]
        assert len(mrns) == 1


class TestRegexDetectorCreditCard:
    """Credit card pattern detection."""

    def test_finds_visa_like_number(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Card: 4111 1111 1111 1111")
        cards = [e for e in result if e.entity_type == "credit_card"]
        assert len(cards) == 1

    def test_finds_dashed_card(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Pay with 4111-1111-1111-1111")
        cards = [e for e in result if e.entity_type == "credit_card"]
        assert len(cards) == 1


class TestRegexDetectorIPAddress:
    """IPv4 address pattern detection."""

    def test_finds_ip_address(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Server at 192.168.1.100 is down")
        ips = [e for e in result if e.entity_type == "ip_address"]
        assert len(ips) == 1
        assert ips[0].text == "192.168.1.100"
        assert ips[0].category == "pci"

    def test_rejects_invalid_ip_octets(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Not an IP: 999.999.999.999")
        ips = [e for e in result if e.entity_type == "ip_address"]
        assert len(ips) == 0


class TestCategoryFiltering:
    """Filtering detection by category."""

    def test_filter_pii_only(self) -> None:
        detector = RegexEntityDetector()
        text = "Email info@test.com, MRN: 12345, IP 10.0.0.1"
        result = detector.detect(text, categories=["pii"])
        assert all(e.category == "pii" for e in result)
        assert any(e.entity_type == "email_address" for e in result)

    def test_filter_pci_only(self) -> None:
        detector = RegexEntityDetector()
        text = "Email info@test.com, IP 10.0.0.1"
        result = detector.detect(text, categories=["pci"])
        assert all(e.category == "pci" for e in result)

    def test_filter_phi_only(self) -> None:
        detector = RegexEntityDetector()
        text = "Email info@test.com, MRN: 12345"
        result = detector.detect(text, categories=["phi"])
        assert all(e.category == "phi" for e in result)
        assert len(result) == 1

    def test_all_keyword_returns_all_categories(self) -> None:
        detector = RegexEntityDetector()
        text = "Email info@test.com, MRN: 12345, IP 10.0.0.1"
        result = detector.detect(text, categories=["all"])
        categories = {e.category for e in result}
        assert len(categories) >= 2

    def test_none_categories_returns_all(self) -> None:
        detector = RegexEntityDetector()
        text = "Email info@test.com, MRN: 12345, IP 10.0.0.1"
        result_all = detector.detect(text, categories=None)
        result_explicit = detector.detect(text, categories=["all"])
        assert len(result_all) == len(result_explicit)

    def test_multiple_categories(self) -> None:
        detector = RegexEntityDetector()
        text = "Email info@test.com, MRN: 12345, IP 10.0.0.1"
        result = detector.detect(text, categories=["pii", "phi"])
        categories = {e.category for e in result}
        assert "pci" not in categories


class TestSortingAndMultipleEntities:
    """Results sorted by start_char, multiple entities in same text."""

    def test_multiple_entities_sorted_by_start_char(self) -> None:
        detector = RegexEntityDetector()
        text = "IP 10.0.0.1 and email user@test.com here"
        result = detector.detect(text)
        assert len(result) >= 2
        for i in range(len(result) - 1):
            assert result[i].start_char <= result[i + 1].start_char

    def test_start_and_end_char_match_text_position(self) -> None:
        detector = RegexEntityDetector()
        text = "Contact user@example.com please"
        result = detector.detect(text, categories=["pii"])
        emails = [e for e in result if e.entity_type == "email_address"]
        assert len(emails) == 1
        entity = emails[0]
        assert text[entity.start_char : entity.end_char] == entity.text


class TestEmptyInput:
    """Edge case: empty or no-match text."""

    def test_empty_text_returns_empty_list(self) -> None:
        detector = RegexEntityDetector()
        assert detector.detect("") == []

    def test_no_entities_returns_empty_list(self) -> None:
        detector = RegexEntityDetector()
        result = detector.detect("Just a normal sentence with nothing special")
        assert result == []


class TestFactory:
    """create_entity_detector() returns a working detector."""

    def test_returns_entity_detector(self) -> None:
        detector = create_entity_detector()
        assert isinstance(detector, EntityDetector)

    def test_returns_regex_detector(self) -> None:
        detector = create_entity_detector()
        assert isinstance(detector, RegexEntityDetector)

    def test_factory_detector_works(self) -> None:
        detector = create_entity_detector()
        result = detector.detect("Email me at test@test.com")
        assert len(result) == 1
        assert result[0].entity_type == "email_address"
