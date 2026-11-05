from google.genai import types


class WebSearchAgent:
    """Find source-backed pages on approved medical websites."""

    def __init__(
        self,
        client_provider,
        url_validator,
        model_name,
        max_sources=5,
    ):
        self.client_provider = client_provider
        self.url_validator = url_validator
        self.model_name = model_name
        self.max_sources = max_sources

    def search(self, question):
        response = self.client_provider().models.generate_content(
            model=self.model_name,
            contents=(
                "Tìm tài liệu y tế trực tiếp liên quan đến câu hỏi. Chỉ ưu "
                "tiên trang chính thức WHO, CDC, NIDDK, ADA hoặc Bộ Y tế Việt "
                "Nam. Trả về các trang nguồn phù hợp, không dùng diễn đàn, "
                "quảng cáo hoặc trang tổng hợp.\n\nCâu hỏi: "
                + question
            ),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0,
                max_output_tokens=512,
            ),
        )

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []

        metadata = getattr(candidates[0], "grounding_metadata", None)
        if metadata is None:
            return []

        chunks = getattr(metadata, "grounding_chunks", None) or []
        supported_indices = set()
        for support in getattr(metadata, "grounding_supports", None) or []:
            supported_indices.update(
                getattr(support, "grounding_chunk_indices", None) or []
            )

        links = []
        seen_urls = set()
        for index, chunk in enumerate(chunks):
            web = getattr(chunk, "web", None)
            url = getattr(web, "uri", None)
            if (
                not url
                or not self.url_validator(url)
                or index not in supported_indices
            ):
                continue
            normalized_url = url.split("#", 1)[0]
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            links.append({
                "url": normalized_url,
                "search_title": getattr(web, "title", None),
            })
            if len(links) >= self.max_sources:
                break

        return links
