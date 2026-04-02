"""
Streamlit UI for the YouTube transcript RAG workflow (same logic as test.ipynb).

Run locally:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from yt_summary import chain_service, rag_service, transcript_service, video_utils


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .main-header {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
            background: linear-gradient(120deg, #ff4b4b 0%, #7c3aed 55%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtle { color: #64748b; font-size: 0.95rem; margin-bottom: 1.25rem; }
        div[data-testid="stExpander"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_openai_key() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    st.error("Set the `OPENAI_API_KEY` environment variable (or add it to a `.env` file and install `python-dotenv`).")
    return False


def main():
    st.set_page_config(
        page_title="TubeTldr — Ask the video",
        page_icon="▶",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.markdown('<p class="main-header">TubeTldr</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Load a YouTube transcript, then ask questions grounded in what was actually said.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        st.caption("Defaults match your notebook: chunk 1000 / overlap 200, retriever k=2, gpt-4o-mini @ 0.2.")
        if not os.environ.get("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY is not set.")

    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "transcript_preview" not in st.session_state:
        st.session_state.transcript_preview = ""

    col_a, col_b = st.columns([3, 1])
    with col_a:
        url_input = st.text_input(
            "YouTube URL or video ID",
            placeholder="https://www.youtube.com/watch?v=… or 11-character ID",
            label_visibility="collapsed",
        )
    with col_b:
        load_clicked = st.button("Load video", type="primary", use_container_width=True)

    if load_clicked:
        if not _ensure_openai_key():
            return
        try:
            vid = video_utils.extract_video_id(url_input)
        except ValueError as e:
            st.error(str(e))
            return
        with st.spinner("Fetching transcript…"):
            text, err = transcript_service.load_transcript_safe(vid)
        if err:
            st.error(err)
            st.session_state.chain = None
            st.session_state.video_id = None
            st.session_state.transcript_preview = ""
            return
        with st.spinner("Embedding transcript and building index…"):
            store = rag_service.build_vector_store(text)
            retriever = rag_service.retriever_from_store(store)
            st.session_state.chain = chain_service.build_main_chain(retriever)
        st.session_state.video_id = vid
        st.session_state.transcript_preview = text[:1200] + ("…" if len(text) > 1200 else "")
        st.success(f"Ready — video `{vid}` indexed.")

    if st.session_state.chain is None:
        st.info("Enter a URL or ID and click **Load video** to build the retriever and chain.")
        with st.expander("How this maps to your notebook"):
            st.markdown(
                """
                1. **Transcript** — `YouTubeTranscriptApi().fetch(...)` → joined text  
                2. **Chunks** — `RecursiveCharacterTextSplitter`  
                3. **FAISS** — `OpenAIEmbeddings` + `FAISS.from_documents`  
                4. **Retriever** — similarity, `k=2`  
                5. **Chain** — `RunnableParallel` → `PromptTemplate` → `ChatOpenAI` → `StrOutputParser`
                """
            )
        return

    st.caption(f"Active video ID: `{st.session_state.video_id}`")

    q1, q2 = st.columns(2)
    with q1:
        if st.button("Quick: summarize the video", use_container_width=True):
            st.session_state["pending_q"] = "Can you summarize the video"
    with q2:
        if st.button("Quick: main themes", use_container_width=True):
            st.session_state["pending_q"] = "What are the main themes discussed in this video?"

    default_q = st.session_state.pop("pending_q", "")
    question = st.text_area(
        "Your question",
        value=default_q,
        height=100,
        placeholder="Ask anything answerable from the transcript…",
    )

    ask = st.button("Ask", type="primary")
    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        elif not _ensure_openai_key():
            return
        else:
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    answer = st.session_state.chain.invoke(question.strip())
                except Exception as e:
                    st.error(f"Model or API error: {e}")
                    return
            st.markdown("### Answer")
            st.write(answer)

    with st.expander("Transcript preview (first ~1200 chars)"):
        st.text(st.session_state.transcript_preview or "(empty)")


main()
