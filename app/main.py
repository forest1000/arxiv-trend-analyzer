import asyncio

import streamlit as st

from arxiv_fetcher import NoveltyAnalyzer, fetch_arxiv_papers


st.set_page_config(
    page_title="arXiv Trend Analyzer with Local GPU LLM",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 GPUローカル要約付き arXivトレンド分析アプリ")
st.markdown(
    """
このアプリは、arXiv.orgに投稿された論文を検索し、**ローカルGPUで動作する要約モデルが各論文の新規性を要約**します。
複数のキーワードを**カンマ（,）**で区切って入力することで、それら全てを含む論文をAND検索できます。
"""
)
st.warning(
    "**注意:** 初回の推論時に要約モデルをダウンロードします。GPUが利用可能な場合は自動的に使用しますが、CPUでも動作します。"
)

st.sidebar.header("検索設定")
search_keywords_input = st.sidebar.text_input(
    "検索キーワード (カンマ区切りでAND検索)",
    "computer vision, object detection, transformer",
    help="""
    複数のキーワードをカンマ（,）で区切って入力してください。\n
    例: `deep learning, medical imaging, segmentation`
    """,
)
max_results = st.sidebar.slider("最大取得件数", 5, 100, 20)

novelty_analyzer = NoveltyAnalyzer()


def _run_fetch(query: str, limit: int):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            fetch_arxiv_papers(query, limit, novelty_analyzer=novelty_analyzer)
        )
    finally:
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()
        asyncio.set_event_loop(None)


if st.sidebar.button("分析を実行"):
    keywords = [k.strip() for k in search_keywords_input.split(",") if k.strip()]

    if not keywords:
        st.warning("検索キーワードを入力してください。")
    else:
        final_query = " AND ".join([f'all:"{k}"' for k in keywords])
        st.sidebar.success("生成されたクエリ:")
        st.sidebar.code(final_query, language="text")

        with st.spinner(
            "論文データを取得し、GPU要約モデルで新規性を分析しています... PCの性能によっては時間がかかります。"
        ):
            try:
                papers = _run_fetch(final_query, max_results)
            except Exception as exc:  # pragma: no cover - runtime guard for UI usage
                st.error(f"処理中にエラーが発生しました: {exc}")
                papers = []

        if papers:
            st.success(f"{len(papers)}件の論文が見つかりました。")

            for i, paper in enumerate(papers):
                st.markdown("---")
                st.subheader(f"{i + 1}. {paper['title']}")
                st.write(f"**著者:** {', '.join(paper['authors'])}")
                st.write(f"**投稿日:** {paper['published_date']}")
                st.write(f"**arXivリンク:** [{paper['url']}]({paper['url']})")

                st.markdown("##### 🤖 要約モデルによる新規性の要約")
                st.info(paper["novelty"])

                with st.expander("元の要旨（Abstract）を読む"):
                    st.write(paper["summary"])
        else:
            st.error("論文が見つかりませんでした。キーワードやネットワーク接続を確認してください。")
else:
    st.info("サイドバーでキーワードを入力し、「分析を実行」ボタンを押してください。")
