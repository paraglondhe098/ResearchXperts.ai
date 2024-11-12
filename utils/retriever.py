import feedparser, requests
import streamlit as st


class ArXivRetriever:
    def __init__(self, api_url="http://export.arxiv.org/api/query?"):
        self.api_url = api_url

    def fetch_papers(self, query, max_results=5):
        query_url = f"{self.api_url}search_query=all:{query}&max_results={max_results}"
        try:
            response = requests.get(query_url)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            papers = []
            for entry in feed.entries:
                paper_id = entry.id.split("/")[-1]
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                paper = {
                    "id": paper_id,
                    "title": entry.title,
                    "category": entry.category,
                    "authors": [author.name for author in entry.authors],
                    "author_comment": entry.get("arxiv_comment", "No comments available"),
                    "published": entry.published,
                    "summary": entry.summary,
                    "link": entry.link,
                    "pdf_url": pdf_url,
                    "journal_ref": entry.get("arxiv_journal_ref", "No journal reference")
                }
                papers.append(paper)
            return papers
        except requests.exceptions.RequestException as e:
            st.error("Error fetching papers. Please try again later.")
            return []
