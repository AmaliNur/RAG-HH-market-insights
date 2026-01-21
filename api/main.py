import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import psycopg
from fastembed import TextEmbedding
from fastapi import FastAPI, Query

# -----------------------------
# Config
# -----------------------------
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

load_dotenv()
DSN = os.getenv("DB_DSN")
if not DSN:
    raise RuntimeError("DB_DSN is not set in .env")

# Глобально держим embedder (быстрее)
_embedder = TextEmbedding(model_name=MODEL)

app = FastAPI(title="HH RAG API", version="0.1.0")


# -----------------------------
# Text utilities
# -----------------------------
def vec_to_pgvector(v) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


RU_STOP = {
    "и","в","во","на","по","к","ко","из","у","за","для","про","под","над","с","со","о","об","от","до","при","как","а","но","или","что","это",
    "мы","вы","они","он","она","оно","я","ты","его","ее","их","наш","ваш",
    "требуется","нужен","нужна","нужны","ищем","работа","вакансия","стажировка","intern","internship","junior","middle","senior"
}
EN_STOP = {
    "and","or","the","a","an","to","for","of","in","on","with","from","as","is","are","be","this","that",
    "job","vacancy","needed","looking"
}


def extract_query_keywords(query: str, max_words: int = 12) -> List[str]:
    q = (query or "").lower()
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


def extract_tech_terms(text: str) -> List[str]:
    t = (text or "").lower()
    found = []
    for p in TECH_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            term = p.replace(r"\b", "").replace("\\", "").strip()
            found.append(term)
    # unique preserve order
    return list(dict.fromkeys(found))


def highlight(text: str, keywords: List[str]) -> str:
    if not keywords:
        return text
    out = text
    for kw in sorted(set(keywords), key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(kw)})\b", r"[\1]", out)
    return out


# -----------------------------
# DB / Retrieval
# -----------------------------
def has_pg_trgm(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("select 1 from pg_extension where extname='pg_trgm'")
        return cur.fetchone() is not None


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


def retrieve_vacancies(
    query: str,
    k: int,
    per_vac: int,
    candidates: int,
    kw_weight: float,
) -> Tuple[List[int], Dict[int, Dict[str, Any]], bool, float]:
    """
    Возвращает:
      - vac_order: список vacancy_id в порядке релевантности
      - vac_map: meta + evidence chunks
      - trgm_available: есть ли pg_trgm
      - kw_weight_used: фактически использованный вес
    """
    q_emb = list(_embedder.embed([query]))[0]
    q_vec = vec_to_pgvector(q_emb)

    with psycopg.connect(DSN) as conn:
        trgm = has_pg_trgm(conn)
        if not trgm:
            kw_weight = 0.0

        with conn.cursor() as cur:
            if trgm and kw_weight > 0:
                cur.execute(
                    SQL_V2_WITH_TRGM,
                    (q_vec, query, q_vec, kw_weight, query, q_vec, candidates, k, per_vac),
                )
            else:
                cur.execute(
                    SQL_V2_NO_TRGM,
                    (q_vec, q_vec, q_vec, candidates, k, per_vac),
                )
            rows = cur.fetchall()

    vac_map: Dict[int, Dict[str, Any]] = {}
    vac_order: List[int] = []

    for r in rows:
        (vac_id, name, comp, area, url, chunk_no, text, dist, kw_sim, score, rn, best_score) = r
        if vac_id not in vac_map:
            vac_map[vac_id] = {
                "vacancy_id": int(vac_id),
                "name": name,
                "employer_name": comp,
                "area_name": area,
                "url": url,
                "best_score": float(best_score),
                "evidence": [],
            }
            vac_order.append(int(vac_id))

        vac_map[vac_id]["evidence"].append(
            {
                "chunk_no": int(chunk_no),
                "text": text,
                "dist": float(dist),
                "kw_sim": float(kw_sim),
                "score": float(score),
                "rank_in_vacancy": int(rn),
            }
        )

    # уникальный порядок (на всякий)
    seen = set()
    vac_order = [x for x in vac_order if not (x in seen or seen.add(x))]
    return vac_order, vac_map, trgm, kw_weight


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health():
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("select 1")
        cur.fetchone()
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(..., min_length=2),
    k: int = 8,
    per_vac: int = 2,
    candidates: int = 250,
    kw_weight: float = 0.25,
    max_quote: int = 800,
    do_highlight: bool = True,
):
    vac_order, vac_map, trgm, kw_used = retrieve_vacancies(
        query=q, k=k, per_vac=per_vac, candidates=candidates, kw_weight=kw_weight
    )
    keywords = extract_query_keywords(q, max_words=12)

    results = []
    for vid in vac_order:
        item = vac_map[vid]
        evidence_out = []
        for ev in item["evidence"]:
            txt = normalize_ws(ev["text"])
            if do_highlight:
                txt = highlight(txt, keywords)
            if len(txt) > max_quote:
                txt = txt[:max_quote] + "..."
            ev2 = dict(ev)
            ev2["text"] = txt
            evidence_out.append(ev2)

        out = dict(item)
        out["evidence"] = evidence_out
        results.append(out)

    return {
        "query": q,
        "hybrid_used": bool(trgm and kw_used > 0),
        "kw_weight_used": float(kw_used),
        "k": k,
        "per_vac": per_vac,
        "results": results,
    }


@app.get("/ask")
def ask(
    q: str = Query(..., min_length=2),
    k: int = 5,
    per_vac: int = 2,
    candidates: int = 250,
    kw_weight: float = 0.25,
    max_quote: int = 700,
    do_highlight: bool = True,
):
    """
    "RAG-ответ" без генерации LLM:
    - берём топ вакансий
    - формируем краткую сводку + reasons
    - даём evidence (цитаты)
    """
    vac_order, vac_map, trgm, kw_used = retrieve_vacancies(
        query=q, k=k, per_vac=per_vac, candidates=candidates, kw_weight=kw_weight
    )
    keywords = extract_query_keywords(q, max_words=12)

    # Tech signals по цитатам
    tech_counter = Counter()
    for vid in vac_order:
        for ev in vac_map[vid]["evidence"]:
            for t in extract_tech_terms(ev["text"]):
                tech_counter[t] += 1
    tech_signals = [t for t, _ in tech_counter.most_common(10)]

    results = []
    for vid in vac_order:
        item = vac_map[vid]

        # reasons
        joined = " ".join([e["text"] for e in item["evidence"]]).lower()
        hit_kw = [kw for kw in keywords if kw in joined][:10]
        hit_tech = extract_tech_terms(joined)[:12]

        why = []
        if hit_kw:
            why.append({"type": "query_match", "items": hit_kw})
        if hit_tech:
            why.append({"type": "tech_terms", "items": hit_tech})

        # evidence formatting
        evidence_out = []
        for ev in item["evidence"]:
            txt = normalize_ws(ev["text"])
            if do_highlight:
                txt = highlight(txt, keywords)
            if len(txt) > max_quote:
                txt = txt[:max_quote] + "..."
            ev2 = dict(ev)
            ev2["text"] = txt
            evidence_out.append(ev2)

        out = dict(item)
        out["why"] = why
        out["evidence"] = evidence_out
        results.append(out)

    # Сводка
    summary_parts = []
    if tech_signals:
        summary_parts.append("Технологии/сигналы: " + ", ".join(tech_signals))
    if keywords:
        summary_parts.append("Ключевые слова запроса: " + ", ".join(keywords))

    summary = {
        "text": " | ".join(summary_parts) if summary_parts else "Результаты сформированы по семантической близости и цитатам.",
        "query_keywords": keywords,
        "tech_signals": tech_signals,
        "notes": [
            "Результаты сгруппированы по вакансиям (не по чанкам).",
            "Цитаты — фрагменты текста вакансий, которые служат доказательствами релевантности (retrieval evidence).",
        ],
    }

    return {
        "query": q,
        "hybrid_used": bool(trgm and kw_used > 0),
        "kw_weight_used": float(kw_used),
        "k": k,
        "per_vac": per_vac,
        "summary": summary,
        "results": results,
    }

@app.get("/stats")
def stats():
    """
    Быстрые метрики по БД и индексу.
    Полезно для демо и контроля пайплайна.
    """
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("select count(*) from vacancies;")
        vacancies_total = cur.fetchone()[0]

        # description заполнен
        cur.execute("""
            select count(*)
            from vacancies
            where description is not null and description <> ''
        """)
        vacancies_with_desc = cur.fetchone()[0]

        cur.execute("select count(*) from vacancy_chunks;")
        chunks_total = cur.fetchone()[0]

        cur.execute("""
            select count(*)
            from vacancy_chunks
            where embedding is not null
        """)
        chunks_embedded = cur.fetchone()[0]

        # pg_trgm включён?
        cur.execute("select exists(select 1 from pg_extension where extname='pg_trgm')")
        trgm_enabled = cur.fetchone()[0]

        # pgvector включён?
        cur.execute("select exists(select 1 from pg_extension where extname='vector')")
        vector_enabled = cur.fetchone()[0]

    return {
        "vacancies_total": int(vacancies_total),
        "vacancies_with_description": int(vacancies_with_desc),
        "chunks_total": int(chunks_total),
        "chunks_with_embedding": int(chunks_embedded),
        "extensions": {
            "pg_trgm": bool(trgm_enabled),
            "vector": bool(vector_enabled),
        },
    }

@app.get("/market/tech-top")
def market_tech_top(limit: int = 20):
    """
    Топ технологий/терминов по корпусу (по текстам чанков).
    Быстро и очень наглядно для демо.
    """
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("select text from vacancy_chunks where text is not null;")
        rows = cur.fetchall()

    counter = Counter()
    for (txt,) in rows:
        for t in extract_tech_terms(txt):
            counter[t] += 1

    top = [{"term": term, "count": int(cnt)} for term, cnt in counter.most_common(limit)]
    return {"top": top, "unique_terms": int(len(counter))}

@app.get("/market/geo")
def market_geo(limit: int = 20):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            select area_name, count(*) as cnt
            from vacancies
            group by area_name
            order by cnt desc
            limit %s
        """, (limit,))
        rows = cur.fetchall()

    top = [{"area_name": a, "count": int(c)} for a, c in rows]
    return {"top": top}

@app.get("/market/employers")
def market_employers(limit: int = 20):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            select employer_name, count(*) as cnt
            from vacancies
            where employer_name is not null and employer_name <> ''
            group by employer_name
            order by cnt desc
            limit %s
        """, (limit,))
        rows = cur.fetchall()

    top = [{"employer_name": e, "count": int(c)} for e, c in rows]
    return {"top": top}

@app.get("/market/keywords")
def market_keywords(limit: int = 30):
    """
    Простейшая частотка токенов по чанкам (без NLP-магии).
    Для демо и quick-insights.
    """
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("select text from vacancy_chunks where text is not null;")
        rows = cur.fetchall()

    cnt = Counter()
    for (txt,) in rows:
        toks = re.findall(r"[a-zа-я0-9\+\#\.]{3,}", (txt or "").lower())
        toks = [t for t in toks if t not in RU_STOP and t not in EN_STOP]
        cnt.update(toks)

    top = [{"token": t, "count": int(c)} for t, c in cnt.most_common(limit)]
    return {"top": top}
