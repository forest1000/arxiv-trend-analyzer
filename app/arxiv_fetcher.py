import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import logging
import asyncio
import ollama 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ARXIV_API_URL = 'http://export.arxiv.org/api/query?'

async def analyze_novelty_with_llm(abstract: str) -> str:
    """
    ローカルで実行されているLLM (Ollama) を使って論文の要旨から新規性を分析・要約する関数

    Args:
        abstract (str): 論文の要旨

    Returns:
        str: LLMによって生成された新規性の要約。エラー時はメッセージを返す。
    """
    prompt = f"""
    Improved Prompt in English
    You are an expert research assistant specializing in the precise analysis of academic papers.
    When evaluating a research paper, it is crucial to distinguish between its "Novelty" and its "Contribution."
    Novelty: Refers to the new ideas, approaches, techniques, or findings that are proposed or demonstrated for the first time in the research. In short, "What is new?"
    Contribution: Refers to the impact and value that the novelty brings to the specific field of study or to society. In short, "How is it useful?"
    Based on these definitions, please carefully read the following abstract and specifically identify the research's "Novelty" and "Contribution."
    #Abstract: {abstract}

    Present your analysis clearly in the following bullet-point format.[Analysis]
    Novelty:
    (Describe the specific points that this research proposes or demonstrates for the first time.)
    Contribution:
    (Describe the value and impact that the novelty brings to the research field or real-world applications.)
    Finally, synthesize the key points of the identified novelty and contribution into a concise summary of approximately 70-100 words.
    [Summary] 
    (Write the summary here.)please only return Novelty, contribution and summary without any additional text in the following format:
    """
    try:
        # Ollamaの非同期クライアントを使用してLLMにリクエストを送信
        response = await ollama.AsyncClient().chat(
            model='deepseek-r1:1.5b',  # 使用するモデルを指定 (例: 'llama3', 'gemma')
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        logging.error(f"Ollamaでの分析中にエラーが発生しました: {e}")

        return "Ollamaでの分析中にエラーが発生しました。Ollamaアプリが起動しているか、指定したモデル (`llama3`など) がダウンロードされているか確認してください。"


async def fetch_arxiv_papers(search_query: str, max_results_per_query: int = 10):
    """
    複数のキーワードで論文を検索し、新規性を分析して結果を返す非同期関数

    Args:
        search_query (str): 検索したいキーワードのリスト
        max_results_per_query (int): 1クエリあたりの最大取得論文数

    Returns:
        list: 論文情報の辞書を含むリスト。エラー時は空リストを返す。
    """
    unique_papers = {}

    query = search_query 
    logging.info(f"'{query}'に関する論文を検索中...")
    params = {
        'search_query': query,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
        'max_results': str(max_results_per_query)
    }
    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespaces = {'arxiv': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('arxiv:entry', namespaces):
            paper_id = entry.find('arxiv:id', namespaces).text
            if paper_id not in unique_papers: 
                title = entry.find('arxiv:title', namespaces).text.strip()
                summary = entry.find('arxiv:summary', namespaces).text.strip().replace('\n', ' ')
                
                unique_papers[paper_id] = {
                    'title': title,
                    'authors': [author.find('arxiv:name', namespaces).text for author in entry.findall('arxiv:author', namespaces)],
                    'summary': summary,
                    'published_date': datetime.strptime(entry.find('arxiv:published', namespaces).text, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d'),
                    'url': paper_id,
                    'novelty': '' 
                }
    except requests.exceptions.RequestException as e:
        logging.error(f"APIリクエストエラー (クエリ: {query}): {e}")
    except ET.ParseError as e:
        logging.error(f"XML解析エラー (クエリ: {query}): {e}")

    if not unique_papers:
        return []
    
    task_to_paper = {
        asyncio.create_task(analyze_novelty_with_llm(paper['summary'])): paper
        for paper in unique_papers.values()
    }
    processed_papers = []

    logging.info(f"{len(unique_papers)}件のユニークな論文の新規性を分析します...")
    
    for future in asyncio.as_completed(task_to_paper.keys()):
        try:
            paper = task_to_paper[future]
            paper = task_to_paper[future]
            novelty_text = await future
            paper['novelty'] = novelty_text
            
            yield paper

        except Exception as e:
            paper = task_to_paper[future]
            paper['novelty'] = f"分析エラー: {e}"

            yield paper




if __name__ == '__main__':
    test_query = "happy"
    results = asyncio.run(fetch_arxiv_papers(search_query=test_query, max_results_per_query=3))
    
    if results:
        for i, paper in enumerate(results):
            print("-" * 50)
            print(f"論文 {i+1}")
            print(f"タイトル: {paper['title']}")
            print(f"URL: {paper['url']}")
            print(f"novelty: {paper['novelty']}")
            print("-" * 50 + "\n")
    else:
        print("テスト実行で論文を取得できませんでした。")