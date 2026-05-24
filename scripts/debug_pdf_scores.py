"""
Debug: compute similarity scores for a text query against all PDF pages.
Run with: conda run -n content-search-ai python scripts/debug_pdf_scores.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./models/mclip_finetuned_coco_ready', device='cpu')

query = 'attention mechanism transformer'
query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype('float32')

conn = sqlite3.connect('content_search_ai.db')
cur = conn.cursor()
cur.execute('SELECT pdf_path, page_number, substr(text_content,1,60), embedding FROM pdf_pages')
rows = cur.fetchall()
conn.close()

results = []
for pdf_path, page_num, snippet, blob in rows:
    vec = np.frombuffer(blob, dtype=np.float32)
    sim = float(np.dot(query_vec, vec))
    results.append((sim, pdf_path, page_num, snippet.replace('\n',' ')))

results.sort(reverse=True)
sims = np.array([s for s,_,_,_ in results])
mean, std = sims.mean(), sims.std()

print(f'Query: "{query}"')
print(f'Total pages: {len(results)}')
print(f'Mean={mean:.4f}  Std={std:.4f}  Min={sims.min():.4f}  Max={sims.max():.4f}')
print(f'Threshold (mean+0.3*std) = {mean+0.3*std:.4f}  -> {(sims>=mean+0.3*std).sum()} pages')
print(f'Threshold (mean+1.5*std) = {mean+1.5*std:.4f}  -> {(sims>=mean+1.5*std).sum()} pages')
print()
print('TOP 10:')
print(f'{"Score":8}  {"PDF":50}  Pg  Snippet')
print('-'*120)
for sim, pdf, pg, snip in results[:10]:
    pdf_short = pdf.replace('data/pdfs/','')[:48]
    print(f'{sim:.4f}    {pdf_short:50}  {pg:3}  {snip[:50]}')
