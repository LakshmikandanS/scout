import os
import requests


TAVILY_API_URL = "https://api.tavily.com/search"


def web_search(query: str, max_results: int = 5) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY environment variable not set")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    response = requests.post(
        TAVILY_API_URL,
        json=payload,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    return [
        {
            "title": result["title"],
            "url": result["url"],
            "content": result["content"],
            "score": result.get("score"),
        }
        for result in data.get("results", [])
    ]


if __name__ == "__main__":
    results = web_search("Python programming")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print("Title:", result["title"])
        print("URL:", result["url"])
        print("Score:", result["score"])
        print("Content:", result["content"])