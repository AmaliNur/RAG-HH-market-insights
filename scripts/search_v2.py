import os
import argparse
from collections import defaultdict

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
    ap.add_argument("--k", type=int, default=8, help="сколько вакансий вернуть")
    ap.add_argument("--per-vac", type=int, default=2, help="сколько цитат (чанков) показать на вакансию")
    ap.add_argument("--candidates", type=int, default=250, help="сколько топ-чанков взять по вектору для перескоринга")
    ap.add_argument("--kw-weight", type=float, default=0.25, help="вес keyword similarity (0 = чисто вектор)")
    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL)
    q_emb = list(model.embed([args.query]))[0]
    q_vec = vec_to_pgvector(q_emb)

    # Идея:
    # 1) берём N лучших чанков по вектору (быстро и широко)
    # 2) считаем keyword similarity (trgm) на этих кандидатах
    # 3) комбинируем score = dist - kw_weight * kw_sim  (меньше = лучше)
    # 4) группируем по vacancy_id и берём 1-2 лучших чанка на вакансию
    # 5) ранжируем вакансии по лучшему score

    sql = """
    with scored as (
      select
        v.vacancy_id, v.name, v.employer_name, v.area_name, v.url,
        c.chunk_no, c.text,
        (c.embedding <=> %s::vector) as dist,
        similarity(lower(c.text), lower(%s)) as kw_sim,
        ((c.embedding <=> %s::vector) - %s * similarity(lower(c.text), lower(%s))) as score
      from vacancy_chunks c
      join vacancies v on v.vacancy_id = c.vacancy_id
      where c.embedding is not null
      order by c.embedding <=> %s::vector
      limit %s
    ),
    ranked as (
      select
        *,
        row_number() over (partition by vacancy_id order by score asc) as rn,
        min(score) over (partition by vacancy_id) as best_score
      from scored
    ),
    top_vacs as (
      select distinct vacancy_id, name, employer_name, area_name, url, best_score
      from ranked
      order by best_score asc
      limit %s
    )
    select
      r.vacancy_id, r.name, r.employer_name, r.area_name, r.url,
      r.chunk_no, r.text, r.dist, r.kw_sim, r.score, r.rn, t.best_score
    from ranked r
    join top_vacs t using (vacancy_id)
    where r.rn <= %s
    order by t.best_score asc, r.rn asc;
    """

    params = (
        q_vec, args.query, q_vec, args.kw_weight, args.query,
        q_vec, args.candidates, args.k, args.per_vac
    )

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Сгруппируем для красивого вывода
    vac_map = defaultdict(lambda: {"meta": None, "chunks": []})
    vac_order = []  # порядок по best_score

    for r in rows:
        vac_id, name, comp, area, url, chunk_no, text, dist, kw_sim, score, rn, best_score = r
        if vac_map[vac_id]["meta"] is None:
            vac_map[vac_id]["meta"] = (name, comp, area, url, best_score)
            vac_order.append(vac_id)
        vac_map[vac_id]["chunks"].append((chunk_no, text, dist, kw_sim, score))

    # Уберём дубли в vac_order (на всякий)
    seen = set()
    vac_order = [x for x in vac_order if not (x in seen or seen.add(x))]

    for i, vac_id in enumerate(vac_order, 1):
        name, comp, area, url, best_score = vac_map[vac_id]["meta"]
        print(f"\n#{i} best_score={best_score:.4f} | {name} — {comp} — {area}")
        if url:
            print(url)

        for (chunk_no, text, dist, kw_sim, score) in vac_map[vac_id]["chunks"]:
            preview = text[:700] + ("..." if len(text) > 700 else "")
            print(f"\n  [chunk {chunk_no}] score={score:.4f} dist={dist:.4f} kw_sim={kw_sim:.4f}")
            print("  " + preview.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
