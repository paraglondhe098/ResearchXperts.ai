from typing import List
from utils.api_keys import fetch_api_key
from dataclasses import dataclass
import os
from sortedcontainers import SortedList
import feedparser
import requests
import streamlit as st
from datetime import datetime


def setup_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = fetch_api_key("langsmith", False)
    os.environ["MISTRAL_API_KEY"] = fetch_api_key("mistral", False)
    os.environ["HF_TOKEN"] = fetch_api_key("HuggingFace", False)


@dataclass
class ResearchPaper:
    """Schema for research paper details"""
    id: str
    title: str
    category: str
    authors: List[str]
    author_comment: str
    published: datetime
    summary: str
    link: str
    pdf_url: str
    journal_ref: str

    @property
    def content(self) -> str:
        """Returns a formatted string representation of the paper"""
        return "\n".join([
            f">>> Id: {self.id}",
            f">>> Title: {self.title}",
            f">>> Category: {self.category}",
            f">>> Authors: {', '.join(self.authors)}",
            f">>> Author Comment: {self.author_comment}",
            f">>> Published: {self.published.strftime('%Y-%m-%d')}",
            f">>> Summary: {self.summary}",
            f">>> Link: {self.link}",
            f">>> PDF URL: {self.pdf_url}",
            f">>> Journal Reference: {self.journal_ref}"
        ])

    def pretty_print(self):
        print(self.content)


class PaperManager:
    """Manages a sorted collection of research papers"""

    def __init__(self):
        self.papers = SortedList(key=lambda paper: paper.published)

    def append(self, paper):
        self.papers.add(paper)

    def __getitem__(self, idx):
        return self.papers[idx]

    def get_papers_by_date_range(self, start_date: str, end_date: str) -> List[ResearchPaper]:
        start = datetime.strptime(start_date, "%m-%Y")
        end = datetime.strptime(end_date, "%m-%Y")

        dummy_start = ResearchPaper(
            id="", title="", category="", authors=[],
            author_comment="", published=start, summary="",
            link="", pdf_url="", journal_ref=""
        )
        dummy_end = ResearchPaper(
            id="", title="", category="", authors=[],
            author_comment="", published=end, summary="",
            link="", pdf_url="", journal_ref=""
        )

        start_idx = self.papers.bisect_left(dummy_start)
        end_idx = self.papers.bisect_right(dummy_end)

        return list(self.papers[start_idx:end_idx])


class ArXivRetriever:
    """Handles retrieval of papers from ArXiv API"""

    def __init__(self, api_url: str = "http://export.arxiv.org/api/query?"):
        self.api_url = api_url

    def fetch_papers(self, query: str, max_results: int = 5):
        query_url = f"{self.api_url}search_query=all:{query}&max_results={max_results}"
        try:
            response = requests.get(query_url, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            papers = PaperManager()
            for entry in feed.entries:
                paper_id = entry.id.split("/")[-1]
                paper = ResearchPaper(
                    id=paper_id,
                    title=entry.title,
                    category=entry.category,
                    authors=[author.name for author in entry.authors],
                    author_comment=entry.get("arxiv_comment", "No comments available"),
                    published=datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ"),
                    summary=entry.summary,
                    link=entry.link,
                    pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
                    journal_ref=entry.get("arxiv_journal_ref", "No journal reference")
                )
                papers.append(paper)
            return papers
        except requests.exceptions.RequestException as e:
            st.error("Error fetching papers. Please try again later.")
            return PaperManager()
