import streamlit as st
import asyncio
from arxiv_fetcher import fetch_arxiv_papers 

# --- ページ設定 ---
st.set_page_config(
    page_title="arXiv Trend Analyzer with Local LLM",
    page_icon="🔬",
    layout="wide"
)

# --- メイン画面 ---
st.title("🔬 ローカルLLM搭載 arXivトレンド分析アプリ")
st.markdown("""
このアプリは、arXiv.orgに投稿された論文を検索し、**あなたのPCで動作するLLMが各論文の新規性を要約**します。
複数のキーワードを**カンマ（,）**で区切って入力することで、それら全てを含む論文をAND検索できます。
""")
st.warning("**注意:** このアプリを使用するには、事前にOllamaをインストールし、バックグラウンドで起動しておく必要があります。")


# --- サイドバー ---
st.sidebar.header("検索設定")
# AND検索用のキーワード入力欄 (text_inputを使用)
search_keywords_input = st.sidebar.text_input(
    "検索キーワード (カンマ区切りでAND検索)",
    'computer vision, object detection, transformer',
    help="""
    複数のキーワードをカンマ（,）で区切って入力してください。\n
    例: `deep learning, medical imaging, segmentation`
    """
)
# 取得件数
max_results = st.sidebar.slider("最大取得件数", 5, 100, 20)

# --- 検索実行と結果表示 ---
if st.sidebar.button("分析を実行"):
    
    keywords = [k.strip() for k in search_keywords_input.split(',') if k.strip()]

    if not keywords:
        st.warning("検索キーワードを入力してください。")
    else:
        # キーワードリストから `all:"keyword1" AND all:"keyword2"` 形式のクエリを生成
        final_query = " AND ".join([f'all:"{k}"' for k in keywords])
        st.sidebar.success("生成されたクエリ:")
        st.sidebar.code(final_query, language='text')

        with st.spinner("論文データを取得し、ローカルLLMで新規性を分析しています... PCの性能によっては時間がかかります。"):
            try:
                # 生成した単一のクエリをリストに入れてfetcherに渡す
                papers = asyncio.run(fetch_arxiv_papers([final_query], max_results))
            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {e}")
                papers = []

        if papers:
            st.success(f"{len(papers)}件の論文が見つかりました。")

            for i, paper in enumerate(papers):
                st.markdown("---")
                st.subheader(f"{i+1}. {paper['title']}")
                st.write(f"**著者:** {', '.join(paper['authors'])}")
                st.write(f"**投稿日:** {paper['published_date']}")
                st.write(f"**arXivリンク:** [{paper['url']}]({paper['url']})")

                st.markdown("##### 🤖 LLMによる新規性の要約")
                st.info(paper['novelty'])

                with st.expander("元の要旨（Abstract）を読む"):
                    st.write(paper['summary'])
        else:
            st.error("論文が見つかりませんでした。キーワードやネットワーク接続を確認してください。")
else:
    st.info("サイドバーでキーワードを入力し、「分析を実行」ボタンを押してください。")

async def output_paper_info(final_query, max_results):
    """
    A function to output paper information asynchronously.
    """
    st.markdown("---")
    st.subheader(paper['title'])
    st.write(f"**著者:** {', '.join(paper['authors'])}")
    st.write(f"**投稿日:** {paper['published_date']}")
    st.write(f"**arXivリンク:** [{paper['url']}]({paper['url']})")

    st.markdown("##### 🤖 LLMによる新規性の要約")
    st.info(paper['novelty'])

    with st.expander("元の要旨（Abstract）を読む"):
        st.write(paper['summary'])