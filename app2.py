import streamlit as st
from datetime import date, timedelta
from utils.core import ArXivRetriever, setup_environment
from utils.assistants import PaperSearchAssistant, PaperAssistant
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma

# Constants
DEFAULT_MAX_RESULTS = 10
MIN_RESULTS = 5
MAX_RESULTS = 50
DEFAULT_FROM_DATE = date.today() - timedelta(days=5*365)  # Last year by default

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'selected_paper': None,
        'papers': None,
        'messages': [],
        'llm': ChatMistralAI(model="mistral-large-2402"),
        'search_history': [],  # New: Keep track of recent searches
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_paper_assistant_response(prompt):
    """Get response from the paper assistant."""
    if not st.session_state.selected_paper:
        return "⚠️ Please select a paper first!"

    if 'assistant' not in st.session_state or st.session_state.assistant.paper.id != st.session_state.selected_paper.id:
        st.session_state.assistant = PaperAssistant(
            paper=st.session_state.selected_paper,
            llm=st.session_state.llm
        )

    response = st.session_state.assistant.answer(prompt)
    return response["answer"]


def display_chat_interface():
    """Display and handle the chat interface."""
    # Display chat messages from history
    st.markdown("### 💬 Ask your questions!")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about the selected paper..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get and display assistant response
        with st.spinner("Thinking..."):
            response = get_paper_assistant_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def display_paper_details(paper, sidebar=False):
    """Display paper details in a formatted way."""
    if sidebar:
        st.header(f"{paper.title}")
    else:
        st.markdown(f"### Title: {paper.title}")
    # st.markdown(f"**Title:** {paper.title}")
    st.markdown("---")
    st.markdown(f"**👥 Authors:** {', '.join(paper.authors)}")
    st.markdown(f"**📅 Published:** {paper.published.strftime('%Y-%m-%d')}")
    st.markdown(f"**🏷️ Category:** {paper.category}")
    if sidebar:
        return
    st.markdown(f"**📝 Summary:** {paper.summary}")
    if paper.journal_ref != "No journal reference":
        st.markdown(f"**📰 Journal Reference:** {paper.journal_ref}")
    if paper.author_comment != "No comments available":
        st.markdown(f"**💭 Author Comments:** {paper.author_comment}")


def display_sidebar():
    """Handle sidebar display and functionality."""
    with st.sidebar:
        if st.session_state.selected_paper:
            st.header("📄 Selected Paper")
            display_paper_details(st.session_state.selected_paper, True)

            if st.button("Clear Selection", type="secondary"):
                st.session_state.selected_paper = None
                st.session_state.messages = []
                st.rerun()

            # PDF viewer
            st.markdown("### 📑 PDF Viewer")
            st.markdown(
                f'<iframe src="{st.session_state.selected_paper.pdf_url}" '
                'width="100%" height="800" type="application/pdf"></iframe>',
                unsafe_allow_html=True
            )
        else:
            st.title("📚 ArXiv Research Assistant")
            st.markdown("Search for papers and chat about them with AI assistance.")


def search_papers(topic, max_results):
    """Execute paper search with error handling."""
    try:
        with st.spinner("Searching papers..."):
            if 'searcher' not in st.session_state:
                st.session_state.searcher = PaperSearchAssistant(llm=st.session_state.llm)
            results = st.session_state.searcher.search_papers(topic, max_results=max_results)

            # Add to search history
            if topic not in st.session_state.search_history:
                st.session_state.search_history = ([topic] + st.session_state.search_history)[:5]

            return results
    except Exception as e:
        st.error(f"⚠️ Error searching papers: {str(e)}")
        return None


def main():
    initialize_session_state()
    display_sidebar()

    # Main content
    st.title("🤖 [ResearchXperts.ai]")

    # Search interface
    col1, col2 = st.columns([1, 1])
    with col1:
        topic = st.text_input("Enter topic or description of the research paper:",
                              placeholder="e.g., machine learning, quantum computing or something random (: ")

        # Show recent searches
        if st.session_state.search_history:
            selected_history = st.selectbox(
                "Recent searches:",
                [""] + st.session_state.search_history,
                index=0
            )
            if selected_history:
                topic = selected_history

    with col2:
        max_results = st.slider(
            "Max results",
            MIN_RESULTS,
            MAX_RESULTS,
            DEFAULT_MAX_RESULTS
        )

    # Date range selection
    col3, col4 = st.columns(2)
    with col3:
        from_date = st.date_input(
            "From date",
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            value=DEFAULT_FROM_DATE
        )
    with col4:
        to_date = st.date_input(
            "To date",
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            value=date.today()
        )

    if st.button("Search Papers", type="primary"):
        if not topic:
            st.warning("Please enter a research topic.")
            return

        results = search_papers(topic, max_results)
        if results:
            st.session_state.papers = results

    # Display results
    if st.session_state.papers:
        st.header("Search Results")

        # Filter papers by date
        filtered_papers = [
            paper for paper in st.session_state.papers.papers
            if from_date <= paper.published.date() <= to_date
        ]

        if not filtered_papers:
            st.info("No papers found in the selected date range.")
            return

        for paper in filtered_papers:
            with st.expander(f"{paper.title} [{paper.published.strftime('%Y-%m-%d')}]"):
                display_paper_details(paper)
                if st.button("📌 Select Paper", key=f"select_{paper.id}"):
                    st.session_state.selected_paper = paper
                    st.session_state.messages = []  # Clear chat when new paper selected
                    st.rerun()

    # Chat interface
    if st.session_state.selected_paper:
        st.markdown("---")
        # st.header("Chat about the Paper")
        display_chat_interface()


if __name__ == "__main__":
    setup_environment()
    main()
