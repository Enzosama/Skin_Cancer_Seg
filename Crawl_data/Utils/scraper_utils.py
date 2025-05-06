import json
import os
import re
from typing import List, Dict
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

from model.SCDataset import SCDataset


def get_browser_config() -> BrowserConfig:
    """
    Returns the browser configuration for the crawler.

    Returns:
        BrowserConfig: The configuration settings for the browser.
    """
    # https://docs.crawl4ai.com/core/browser-crawler-config/
    return BrowserConfig(
        browser_type="chromium",  # Type of browser to simulate
        headless=False,  # No GUI
        verbose=True,  # Enable verbose logging
    )


def get_llm_strategy() -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.

    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        provider="groq/deepseek-r1-distill-llama-70b",
        api_token=os.getenv("GROQ_API_KEY"),
        schema=SCDataset.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "From the Markdown content, extract fields: disease_name, article title, questions, and corresponding answers. "
            "Convert bolded section titles into questions by using the title text as the question. "
            "Use the content immediately following each bolded section as the answer. "
            "Exclude any 'Authors', 'Advertisement', and 'References' sections, and strip out any URLs, ads, phone numbers, or extra info."
        ),
        input_format="markdown",
        verbose=True,
    )


def clean_text(text: str) -> str:
    # remove markdown links and HTML tags
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Preserve paragraphs: strip extra spaces but keep blank lines
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            # keep only one blank line between paragraphs
            if not prev_blank:
                cleaned.append('')
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return '\n'.join(cleaned).strip()


def parse_markdown(md: str, url: str, disease_name: str) -> list:
    import re
    title = ""
    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    rows = []
    q = None
    ans_lines = []

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Bỏ qua các dòng không cần thiết
        if any(token in line for token in ["http://", "https://", "Advertisement", "Authors", "References", "javascript:"]):
            i += 1
            continue

        # Nhận diện tiêu đề (H1, H2, H3, **bold**)
        bold_match = re.match(r'^(#{2,6})\s+(.+)', line)
        strong_match = re.match(r'^\*\*(.+?)\*\*$', line)
        if bold_match or strong_match:
            # Lưu Q&A trước đó nếu có
            # Include blank lines to preserve paragraphs
            raw_answer = "\n".join(ans_lines).strip()
            answer = clean_text(raw_answer)
            if q is not None and is_relevant_qa(q, answer):
                rows.append({
                    "url": url,
                    "disease_name": disease_name,
                    "title": title,
                    "question": q,
                    "answer": answer,
                })
            # Lấy question mới
            if bold_match:
                q = clean_text(bold_match.group(2).strip())
            elif strong_match:
                q = clean_text(strong_match.group(1).strip())
            ans_lines = []
            i += 1
            continue

        # Skip H1 lines (already used for article title)
        if line.startswith("# "):
            i += 1
            continue

        # Thu thập answer: lấy cả dòng văn và bullet (bắt đầu bằng *, - hoặc +)
        if q is not None:
            # Giữ nguyên các bullet hoặc dòng văn bản
            ans_lines.append(line)
        i += 1

    # Lưu Q&A cuối cùng
    # Include blank lines to preserve paragraphs
    raw_answer = "\n".join(ans_lines).strip()
    answer = clean_text(raw_answer)
    if q is not None and answer and is_relevant_qa(q, answer):
        rows.append({
            "url": url,
            "disease_name": disease_name,
            "title": title,
            "question": q,
            "answer": answer,
        })

    return rows


def is_relevant_qa(question: str, answer: str) -> bool:
    """
    Trả về True nếu QA này là nội dung mong muốn, False nếu là nội dung phụ/trang web.
    """
    unwanted_questions = [
        "Search", "Quick links", "About the journal", "Explore content", "Publish with us",
        "Similar content being viewed by others", "Quality assurance", "General", "Fig. 1",
        "Background & Summary", "Find Us On", "Membership", "Editions", "All material on this website is protected by copyright",
        "References", "Medically Reviewed", "Last reviewed", "Learn more about the Health Library", "editorial process",
        "Media Gallery", "Contributor", "Disclosure",
        "Previous", "Next"
    ]
    # Loại bỏ các câu hỏi hoặc answer chứa các cụm không mong muốn (dùng lower để so khớp không phân biệt hoa thường)
    for unwanted in unwanted_questions:
        if unwanted.lower() in question.strip().lower() or unwanted.lower() in answer.strip().lower():
            return False
    # Loại bỏ answer hoặc question quá ngắn
    if len(question.strip()) < 3 or len(answer.strip()) < 3:
        return False
    # Filter UI artifacts from scraped tables, checkboxes, buttons
    artifacts = ["checkbox", "filter button", "apply cancel", "confirm my choices", "allow all"]
    q_lower = question.strip().lower()
    a_lower = answer.strip().lower()
    for token in artifacts:
        if token in q_lower or token in a_lower:
            return False
    # Filter phone numbers or call prompts
    if re.search(r"\b\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}\b", answer):
        return False
    if re.search(r"call\s*\d", a_lower):
        return False
    return True