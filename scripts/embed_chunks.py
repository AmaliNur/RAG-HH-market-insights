import os, argparse
from dotenv import load_dotenv
import psycopg
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim (multilingual) :contentReference[oaicite:2]{index=2}

def vec_to_pgvector(v):
    # v может быть list/np.array; приводим к строке вида [0.1,0.2,...]
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"

def main():
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("DB_DSN is not set in .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = без лимита")
    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL)

    done = 0
    with psycopg.connect(dsn) as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, text
                    from vacancy_chunks
                    where embedding is null
                    order by id
                    limit %s
                    """,
                    (args.batch,),
                )
                rows = cur.fetchall()

            if not rows:
                break

            ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]

            embs = list(model.embed(texts))

            with conn.cursor() as cur:
                for chunk_id, emb in zip(ids, embs):
                    cur.execute(
                        "update vacancy_chunks set embedding=%s::vector where id=%s",
                        (vec_to_pgvector(emb), chunk_id),
                    )
                conn.commit()

            done += len(rows)
            print("embedded:", done)

            if args.limit and done >= args.limit:
                break

    print("DONE embedded:", done)

if __name__ == "__main__":
    main()
