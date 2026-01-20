import os, argparse
from dotenv import load_dotenv
import psycopg
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def vec_to_pgvector(v):
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"

def main():
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("DB_DSN is not set in .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL)
    q_emb = list(model.embed([args.query]))[0]
    q_vec = vec_to_pgvector(q_emb)

    sql = """
    select
      v.vacancy_id, v.name, v.employer_name, v.area_name, v.url,
      c.chunk_no, c.text,
      (c.embedding <=> %s::vector) as dist
    from vacancy_chunks c
    join vacancies v on v.vacancy_id = c.vacancy_id
    where c.embedding is not null
    order by c.embedding <=> %s::vector
    limit %s;
    """

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (q_vec, q_vec, args.k))
        rows = cur.fetchall()

    for i, r in enumerate(rows, 1):
        vac_id, name, comp, area, url, chunk_no, text, dist = r
        print(f"\n#{i} dist={dist:.4f} | {name} — {comp} — {area}")
        print(url or "")
        print(f"[chunk {chunk_no}]\n{text[:700]}{'...' if len(text) > 700 else ''}")

if __name__ == "__main__":
    main()
