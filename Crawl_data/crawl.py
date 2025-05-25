import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from model.SCDataset import SCDataset
from Utils.scraper_utils import parse_markdown
import os, csv
import sys
import re

###Set up to calling groq model llm 
'''
curl https://api.groq.com/openai/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer Api_Key" \
-d '{
"model": "meta-llama/llama-4-scout-17b-16e-instruct",
"messages": [{
    "role": "user",
    "content": "Explain the importance of fast language models"
}]
}'
'''


async def main():
    browser_config = BrowserConfig()  
    run_config = CrawlerRunConfig()   

    # Load disease names and URLs from CSV
    urls = []
    import csv as _csv
    with open("web_dataset.csv", newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                urls.append((row[0].strip(), row[1].strip()))
    # prepare output CSV
    out_file = "rag.csv"
    write_header = not os.path.exists(out_file)
    fout = open(out_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["disease_name", "url", "title", "question", "answer"])

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for disease_name, url in urls:
            print(f"Fetching {url} for {disease_name}...")
            try:
                result = await crawler.arun(url=url, config=run_config)
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue
            md = result.markdown or ""
            # parse Q&A entries
            rows = parse_markdown(md, url, disease_name)
            # Helpers
            def rephrase_question(q, dn):
                if not q or dn.lower() not in q.lower():
                    return f"{q.strip()} ({dn})" if q else f"What is {dn}?"
                return q
            def verify_entry(entry, dn):
                txt = (entry.get('answer','') + ' ' + entry.get('question','')).lower()
                return dn.lower() in txt
            valid = []
            for entry in rows:
                entry['question'] = rephrase_question(entry.get('question',''), disease_name)
                if verify_entry(entry, disease_name):
                    valid.append(entry)
                else:
                    print(f"Skipping entry for {disease_name} at {url}: not relevant", file=sys.stderr)
            for entry in valid:
                writer.writerow([entry["disease_name"], entry["url"], entry["title"], entry["question"], entry["answer"]])
            print(f"Wrote {len(valid)} valid entries for {url} (out of {len(rows)})")
    fout.close()

if __name__ == "__main__":
    asyncio.run(main())
