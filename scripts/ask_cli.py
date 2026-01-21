import os
import re
import argparse
from collections import Counter, defaultdict

from dotenv import load_dotenv
import psycopg
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# -----------------------------
# Helpers
# -----------------------------
def vec_to_pgvector(v) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


RU_STOP = {
    "и","в","во","на","по","к","ко","из","у","за","для","про","под","над","с","со","о","об","от","до","при","как","а","но","или","что","это",
    "мы","вы","они","он","она","оно","я","ты","его","ее","их","наш","ваш",
    "требуется","нужен","нужна","нужны","ищем","работа","вакансия","стажировка","junior","middle","senior"
}
EN_STOP = {
    "and","or","the","a","an","to","for","of","in","on","with","from","as","is","are","be","this","that",
    "job","vacancy","intern","internship","needed","looking"
}


def extract_query_keywords(query: str, max_words: int = 10):
    """
    Очень простая выжимка ключевых слов из запроса:
    - токены >= 3 символов
    - убираем стоп-слова
    - оставляем топ по частоте
    """
    q = query.lower()
    tokens = re.findall(r"[a-zа-я0-9\+\#\.]{3,}", q, flags=re.IGNORECASE)
    tokens = [t for t in tokens if t not in RU_STOP and t not in EN_STOP]
    cnt = Counter(tokens)
    return [w for w, _ in cnt.most_common(max_words)]


TECH_PATTERNS = [
    r"\bjava\b", r"\bspring\b", r"\bspring boot\b", r"\bpython\b", r"\bgo\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bsql\b", r"\bpostgres\b", r"\bpostgresql\b", r"\bclickhouse\b", r"\bmysql\b", r"\bmongodb\b",
    r"\bairflow\b", r"\bkafka\b", r"\bredis\b", r"\bkubernetes\b", r"\bdocker\b", r"\bterraform\b",
    r"\bhadoop\b", r"\bspark\b", r"\bflink\b", r"\bdatabricks\b",
    r"\betl\b", r"\belt\b", r"\bdwh\b", r"\bmlops\b", r"\bci\/cd\b",
]


def extract_tech_terms(text: str):
    t = (text or "").lower()
    found = []
    for p in TECH_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            # нормализуем “spring boot” и т.п.
            term = p.replace(r"\b", "").replace("\\", "").strip()
            found.append(term)
    return found


def highlight(text: str, keywords):
    """Подсветка ключевых слов в цитатах (ненавязчиво, в консоли)."""
    if not keywords:
        return text
    out = text
    # длинные сначала, чтобы spring boot подсветился до spring
    for kw in sorted(set(keywords), key=len, reverse=True):
        out = re.sub(
            rf"(?i)\b({re.escape(kw)})\b",
            r"[\1]",
            out
        )
    return out


def has_pg_trgm(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("select 1 from pg_extension where extname='pg_trgm'")
        return cur.fetchone() is not None


# -----------------------------
# Retrieval (vacancy-level)
# -----------------------------
SQL_V2_WITH_TRGM = """
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

SQL_V2_NO_TRGM = """
with scored as (
  select
    v.vacancy_id, v.name, v.employer_name, v.area_name, v.url,
    c.chunk_no, c.text,
    (c.embedding <=> %s::vector) as dist,
    0.0 as kw_sim,
    (c.embedding <=> %s::vector) as score
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


def retrieve(conn, query: str, q_vec: str, k: int, per_vac: int, candidates: int, kw_weight: float):
    trgm = has_pg_trgm(conn)
    if not trgm:
        kw_weight = 0.0

    with conn.cursor() as cur:
        if trgm:
            cur.execute(
                SQL_V2_WITH_TRGM,
                (q_vec, query, q_vec, kw_weight, query, q_vec, candidates, k, per_vac)
            )
        else:
            cur.execute(
                SQL_V2_NO_TRGM,
                (q_vec, q_vec, q_vec, candidates, k, per_vac)
            )
        rows = cur.fetchall()

    vac_map = defaultdict(lambda: {"meta": None, "chunks": []})
    vac_order = []
    for r in rows:
        vac_id, name, comp, area, url, chunk_no, text, dist, kw_sim, score, rn, best_score = r
        if vac_map[vac_id]["meta"] is None:
            vac_map[vac_id]["meta"] = (name, comp, area, url, best_score)
            vac_order.append(vac_id)
        vac_map[vac_id]["chunks"].append((chunk_no, text, dist, kw_sim, score))

    # уникальный порядок
    seen = set()
    vac_order = [x for x in vac_order if not (x in seen or seen.add(x))]
    return vac_order, vac_map, trgm


# -----------------------------
# "Ask" (compose answer)
# -----------------------------
def main():
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("DB_DSN is not set in .env")

    ap = argparse.ArgumentParser(description="RAG ask (console): top vacancies + evidence quotes")
    ap.add_argument("query", type=str, help="пользовательский запрос")
    ap.add_argument("--k", type=int, default=5, help="сколько вакансий вернуть")
    ap.add_argument("--per-vac", type=int, default=2, help="сколько цитат на вакансию")
    ap.add_argument("--candidates", type=int, default=250, help="кандидаты по вектору для перескоринга")
    ap.add_argument("--kw-weight", type=float, default=0.25, help="вес keyword similarity (если есть pg_trgm)")
    ap.add_argument("--max-quote", type=int, default=700, help="макс длина цитаты в выводе")
    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL)
    q_emb = list(model.embed([args.query]))[0]
    q_vec = vec_to_pgvector(q_emb)

    keywords = extract_query_keywords(args.query, max_words=12)

    with psycopg.connect(dsn) as conn:
        vac_order, vac_map, trgm = retrieve(
            conn=conn,
            query=args.query,
            q_vec=q_vec,
            k=args.k,
            per_vac=args.per_vac,
            candidates=args.candidates,
            kw_weight=args.kw_weight,
        )

    # Соберём общий “срез” по тех. словам (по цитатам)
    tech_counter = Counter()
    for vac_id in vac_order:
        for (_, text, *_rest) in vac_map[vac_id]["chunks"]:
            for term in extract_tech_terms(text):
                tech_counter[term] += 1

    # ----------------- Output -----------------
    print("\n" + "=" * 80)
    print("RAG ANSWER (console)")
    print("=" * 80)
    print(f"Query: {args.query}")
    if keywords:
        print("Query keywords:", ", ".join(keywords))
    print("Hybrid:", "vector + trgm" if trgm and args.kw_weight > 0 else "vector only")
    if tech_counter:
        top_terms = [t for t, _ in tech_counter.most_common(8)]
        print("Tech signals (top):", ", ".join(top_terms))
    print("-" * 80)

    if not vac_order:
        print("No results. Попробуй другой запрос или увеличь корпус/чанки.")
        return

    for i, vac_id in enumerate(vac_order, 1):
        name, comp, area, url, best_score = vac_map[vac_id]["meta"]
        print(f"\n#{i} | best_score={best_score:.4f}")
        print(f"{name} — {comp} — {area}")
        if url:
            print(url)

        # “Почему подходит” — очень простая логика: упоминания ключевых слов/техов в цитатах
        reasons = []
        vac_text_all = " ".join([c[1] for c in vac_map[vac_id]["chunks"]]).lower()

        hit_kw = [kw for kw in keywords if kw in vac_text_all]
        if hit_kw:
            reasons.append("совпадения по запросу: " + ", ".join(hit_kw[:8]))

        hit_tech = []
        for term in extract_tech_terms(vac_text_all):
            hit_tech.append(term)
        hit_tech = list(dict.fromkeys(hit_tech))  # unique keep order
        if hit_tech:
            reasons.append("технологии: " + ", ".join(hit_tech[:10]))

        if reasons:
            print("Почему подходит:")
            for r in reasons:
                print("  - " + r)

        print("Доказательства (цитаты):")
        for (chunk_no, text, dist, kw_sim, score) in vac_map[vac_id]["chunks"]:
            t = normalize_ws(text)
            t = highlight(t, keywords)
            if len(t) > args.max_quote:
                t = t[:args.max_quote] + "..."

            print(f"  • chunk {chunk_no} | score={score:.4f} dist={dist:.4f} kw={kw_sim:.4f}")
            print("    " + t)

    print("\n" + "=" * 80)
    print("Tip: попробуй разные запросы и сравнивай выдачу до/после enrichment/hybrid.")
    print("=" * 80)


if __name__ == "__main__":
    main()
