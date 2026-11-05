from src.agents.support_agent import SupportAgent


def test_knowledge_summary_includes_answer_and_cited_sources():
    response = SupportAgent()._format_knowledge({
        "success": True,
        "answer": "Một câu trả lời có dẫn chứng [1].",
        "answer_mode": "generated",
        "results": [{
            "title": "Diabetes basics",
            "organization": "CDC",
            "url": "https://www.cdc.gov/diabetes/about/index.html",
            "content": "Relevant evidence.",
        }],
        "sources": [{
            "title": "Diabetes basics",
            "organization": "CDC",
            "url": "https://www.cdc.gov/diabetes/about/index.html",
            "content": "Relevant evidence.",
            "citation_ids": [1, 2],
        }],
    })

    text = "\n".join(response)
    assert "Một câu trả lời có dẫn chứng [1]." in text
    assert "[1, 2] Diabetes basics (CDC)" in text
    assert "https://www.cdc.gov/diabetes/about/index.html" in text
    assert "Relevant evidence." not in text


def test_knowledge_summary_preserves_evidence_only_answer():
    response = SupportAgent()._format_knowledge({
        "success": True,
        "answer": "Gemini hiện không khả dụng.\n\n[1] Trích đoạn nguồn.",
        "answer_mode": "evidence_only",
        "results": [{
            "title": "Diabetes basics",
            "url": "https://www.cdc.gov/diabetes/about/index.html",
        }],
        "sources": [{
            "title": "Diabetes basics",
            "url": "https://www.cdc.gov/diabetes/about/index.html",
        }],
    })

    text = "\n".join(response)
    assert "Gemini hiện không khả dụng." in text
    assert "[1] Trích đoạn nguồn." in text


def test_knowledge_summary_shows_no_evidence_message():
    response = SupportAgent()._format_knowledge({
        "success": True,
        "answer": None,
        "results": [],
        "sources": [],
        "message": "Chưa tìm thấy nguồn y tế phù hợp.",
    })

    assert "Chưa tìm thấy nguồn y tế phù hợp." in "\n".join(response)
