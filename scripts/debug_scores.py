"""
Debug: compute similarity scores for a test query against all images.
Run with: conda run -n content-search-ai python scripts/debug_scores.py
"""
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from core.image_search import build_prompt_variants

mclip = SentenceTransformer('./models/mclip_finetuned_coco_ready', device='cpu')

query = 'politician in suit'
prompts = build_prompt_variants(query)
q_embs = mclip.encode(prompts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')

conn = sqlite3.connect('content_search_ai.db')
cur = conn.cursor()
cur.execute('SELECT filename, embedding FROM images')
rows = cur.fetchall()
conn.close()

results = []
for fname, blob in rows:
    vec = np.frombuffer(blob, dtype=np.float32)
    norm = np.linalg.norm(vec)
    sim = max(float(np.dot(q, vec)) for q in q_embs)
    results.append((sim, norm, fname))

results.sort(reverse=True)

print(f'\nTOP 15 results for: "{query}"')
print(f'{"Score":8}  {"ImgNorm":8}  Filename')
print('-'*80)
for sim, norm, fname in results[:15]:
    print(f'{sim:.4f}    {norm:.4f}    {fname}')

sims = np.array([s for s,_,_ in results])
mean, std = sims.mean(), sims.std()
print(f'\nStats over {len(results)} images:')
print(f'  Mean={mean:.4f}  Std={std:.4f}  Min={sims.min():.4f}  Max={sims.max():.4f}')
print(f'  Threshold (mean+0.3*std) = {mean+0.3*std:.4f}  -> passes {(sims >= mean+0.3*std).sum()} images')
print(f'  Threshold (mean+1.0*std) = {mean+1.0*std:.4f}  -> passes {(sims >= mean+1.0*std).sum()} images')
print(f'  Threshold (mean+1.5*std) = {mean+1.5*std:.4f}  -> passes {(sims >= mean+1.5*std).sum()} images')
print(f'  Absolute 0.28            = 0.2800  -> passes {(sims >= 0.28).sum()} images')

print(f'\nQuery embedding norms: {[round(np.linalg.norm(q),4) for q in q_embs[:3]]}')
