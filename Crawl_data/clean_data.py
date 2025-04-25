import os
import csv
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def call_groq_llama(question, answer, disease_name, model="llama-3.3-70b-versatile"):
    client = Groq()
    prompt = (
        f"Bạn là chuyên gia y tế. Hãy xác định liệu câu trả lời sau có phù hợp, liên quan và đúng với câu hỏi và chủ đề bệnh '{disease_name}' không. "
        f"Nếu không hợp lệ, trả lời 'INVALID'. Nếu hợp lệ, trả lời 'VALID'.\n"
        f"Câu hỏi: {question}\n"
        f"Câu trả lời: {answer}\n"
        f"Chủ đề bệnh: {disease_name}\n"
        f"Chỉ trả lời 'VALID' hoặc 'INVALID'."
    )
    print(f"[DEBUG] Gọi model: {model}\nPrompt: {prompt}")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y tế về chủ đề bệnh học."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        result = completion.choices[0].message.content.strip()
        print(f"[DEBUG] Kết quả model: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] {e}")
        return "INVALID"

import json

def clean_csv(input_csv, output_txt):
    valid_rows = []
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            disease_name = row['disease_name']
            question = row['question']
            answer = row['answer']
            title = row.get('title', '')
            # Gọi API để kiểm tra tính hợp lệ với question
            try:
                result = call_groq_llama(question, answer, disease_name)
            except Exception as e:
                print(f"API error: {e}")
                result = "INVALID"
            print(f"Q: {question}\nA: {answer}\nResult: {result}\n---")
            # Nếu INVALID với question, thử lại với title
            if result == "INVALID" and title:
                print(f"[DEBUG] Thử lại với title: {title}")
                try:
                    result_title = call_groq_llama(title, answer, disease_name)
                except Exception as e:
                    print(f"API error (title): {e}")
                    result_title = "INVALID"
                print(f"Title: {title}\nA: {answer}\nResult (title): {result_title}\n---")
                if result_title == "VALID":
                    valid_rows.append(row)
                    time.sleep(1.2)
                    continue
                result = result_title  # để logic sau chỉ ghi nếu VALID
            if result == "VALID":
                valid_rows.append(row)
            time.sleep(1.2)  # Tránh rate limit
    # Ghi ra file clean_rag.txt toàn bộ dòng hợp lệ
    with open(output_txt, 'w', encoding='utf-8') as txtfile:
        for row in valid_rows:
            txtfile.write(json.dumps(row, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_csv = os.path.join(os.path.dirname(__file__), 'rag.csv')
    output_txt = os.path.join(os.path.dirname(__file__), 'clean_rag.txt')
    clean_csv(input_csv, output_txt)
    print(f"Đã lọc xong. File kết quả: {output_txt}")
