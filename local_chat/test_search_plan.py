"""Natural-language search constraint tests."""
from local_chat import search_plan


def test_persian_language_constraint_is_not_misread_as_single_site():
    plan = search_plan.parse_search_request("قیمت RTX 5070 فقط سایت‌های فارسی")
    assert plan["constraints"]["language"] == "fa"
    assert plan["strict_domains"] is False
    assert plan["query"] == "قیمت RTX 5070"


def test_named_iranian_site_becomes_strict_domain_scope():
    plan = search_plan.parse_search_request("قیمت RTX 5070 فقط سایت دیجی‌کالا")
    assert plan["constraints"]["allowed_domains"] == ["digikala.com"]
    assert plan["strict_domains"] is True
    assert plan["query"] == "قیمت RTX 5070"


def test_malformed_domain_does_not_crash_search_plan():
    plan = search_plan.parse_search_request(
        "موضوع",
        {"allowed_domains": ["https://[invalid", "javascript:alert(1)"]},
    )
    assert plan["constraints"]["allowed_domains"] == []


def test_explicit_domain_allow_and_deny_lists():
    allowed = search_plan.parse_search_request("benchmark RTX 5070 فقط سایت example.com")
    assert allowed["constraints"]["allowed_domains"] == ["example.com"]
    assert search_plan.host_allowed("shop.example.com", allowed["constraints"])
    assert not search_plan.host_allowed("other.com", allowed["constraints"])

    excluded = search_plan.parse_search_request("benchmark RTX 5070 به جز example.com")
    assert not search_plan.host_allowed("shop.example.com", excluded["constraints"])
    assert search_plan.host_allowed("other.com", excluded["constraints"])


def test_allowed_and_excluded_sites_can_be_combined():
    plan = search_plan.parse_search_request(
        "قیمت RTX 5070 فقط سایت example.com به جز bad.example"
    )
    assert plan["constraints"]["allowed_domains"] == ["example.com"]
    assert plan["constraints"]["excluded_domains"] == ["bad.example"]


def test_site_alias_without_word_site_is_understood_and_removed_from_query():
    plan = search_plan.parse_search_request("قیمت RTX 5070 فقط دیجی‌کالا")
    assert plan["constraints"]["allowed_domains"] == ["digikala.com"]
    assert plan["strict_domains"] is True
    assert plan["query"] == "قیمت RTX 5070"


def test_persian_sites_filter_does_not_require_a_domain():
    plan = search_plan.parse_search_request("مقاله هوش مصنوعی فقط سایت‌های فارسی")
    assert plan["constraints"]["language"] == "fa"
    assert plan["strict_domains"] is False


def test_ai_topic_is_not_mistaken_for_an_ai_file_extension():
    plan = search_plan.parse_search_request("مقاله درباره AI و یادگیری ماشین")
    assert plan["constraints"]["content_types"] == []
    assert plan["constraints"]["file_search"] is False
    file_plan = search_plan.parse_search_request("فایل درباره AI و یادگیری ماشین")
    assert file_plan["constraints"]["content_types"] == []


def test_generic_file_search_marks_direct_file_results():
    plan = search_plan.parse_search_request("فایل‌های مربوط به آموزش پایتون را پیدا کن")
    assert plan["constraints"]["file_search"] is True
    assert plan["constraints"]["content_types"] == []


def test_ordinary_site_mention_is_preserved_as_search_content():
    plan = search_plan.parse_search_request("ویدیو آموزش پایتون در آپارات پیدا کن")
    assert "آپارات" in plan["query"]
    assert plan["constraints"]["allowed_domains"] == []


def test_file_filter_requires_a_direct_file_url():
    constraints = {"file_search": True, "content_types": ["pdf"]}
    assert search_plan.result_matches_constraints(
        {"url": "https://example.com/report.pdf", "title": "Report", "snippet": ""},
        constraints,
    )
    assert not search_plan.result_matches_constraints(
        {"url": "https://example.com/report", "title": "Report", "snippet": ""},
        constraints,
    )


def test_file_type_is_extracted_without_llm():
    plan = search_plan.parse_search_request("مقاله یادگیری ماشین فایل PDF")
    assert "pdf" in plan["constraints"]["content_types"]
    assert "filetype:pdf" in plan["query"] or plan["query"] == "مقاله یادگیری ماشین فایل PDF"


def test_model_supplied_search_plan_is_sanitized():
    plan = search_plan.parse_search_request(
        "بحث درباره مدل‌ها",
        {"allowed_domains": ["https://example.com/path", "javascript:alert(1)"],
         "language": "fa", "content_types": ["pdf", "exe"]},
    )
    assert plan["constraints"]["allowed_domains"] == ["example.com"]
    assert plan["constraints"]["content_types"] == ["pdf", "exe"]
