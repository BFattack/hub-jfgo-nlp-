import re
import io

file_path = "news.txt"
doc_text = file_path.read()

processed_text = doc_text.strip()

sentences = re.split(r'(?<=[。！？])\s*', processed_text)

sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

cleaned_text = "\n".join(sentences)

with open("output.txt","w") as f:
  f.write(cleaned_txt)
  
