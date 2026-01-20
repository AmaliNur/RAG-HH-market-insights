import os, re, html, json, argparse
from dotenv import load_dotenv
import psycopg

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)          # убираем HTML / highlighttext
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_doc(row) -> str:
    vacancy_id, name, employer, area, published_at, description, raw, url = row

    # raw может прийти dict или строкой
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}

    snippet = (raw or {}).get("snippet") or {}
    req = clean_text(snippet.get("requirement", ""))
    resp = clean_text(snippet.get("responsibility", ""))

    parts = [
        f"Vacancy: {name or ''}",
        f"Company: {employer or ''}",
        f"Location: {area or ''}",
        f"Published: {published_at or ''}",
    ]

    desc = clean_text(description or "")
    if desc:
        parts.append("Description: " + desc)
    else:
        if req:
            parts.append("Requirements: " + req)
        if resp:
            parts.append("Responsibilities: " + resp)

    return "\n".join([p for p in parts if p.strip()])

def chunk_text(text: str, max_chars: int, overlap: int):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end].strip())
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def main():
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("DB_DSN is not set in .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--rebuild", type=int, default=0, help="1 = пересобрать чанки заново (удалит старые)")
    args = ap.parse_args()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from vacancies")
            total = cur.fetchone()[0]
            print("vacancies:", total)

        with conn.cursor() as cur:
            cur.execute("""
                select vacancy_id, name, employer_name, area_name, published_at, description, raw, url
                from vacancies
                order by vacancy_id
            """)
            rows = cur.fetchall()

        inserted = 0
        with conn.cursor() as cur:
            for row in rows:
                vacancy_id = row[0]
                doc = build_doc(row)
                chunks = chunk_text(doc, args.max_chars, args.overlap)

                if args.rebuild:
                    cur.execute("delete from vacancy_chunks where vacancy_id=%s", (vacancy_id,))

                for i, ch in enumerate(chunks):
                    cur.execute(
                        """
                        insert into vacancy_chunks(vacancy_id, chunk_no, text, source_url)
                        values (%s,%s,%s,%s)
                        on conflict (vacancy_id, chunk_no) do update
                          set text=excluded.text, source_url=excluded.source_url
                        """,
                        (vacancy_id, i, ch, row[7]),
                    )
                    inserted += 1

            conn.commit()

    print("chunks upserted:", inserted)

if __name__ == "__main__":
    main()
