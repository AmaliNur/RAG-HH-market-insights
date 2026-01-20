import os, time, json, argparse
from pathlib import Path
import requests
import psycopg
from tqdm import tqdm
from dotenv import load_dotenv

API = "https://api.hh.ru"

def fetch_json(url, headers, params=None, max_retries=6):
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=40)

        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "2"))
            time.sleep(wait)
            continue

        if r.status_code >= 500:
            time.sleep(1 + attempt)
            continue

        if r.status_code >= 400:
            try:
                print("HH error:", r.status_code, r.json())
            except Exception:
                print("HH error:", r.status_code, r.text[:500])
            r.raise_for_status()

        r.raise_for_status()
        return r.json()

    raise RuntimeError(f"Failed after retries: {url}")

def upsert_vacancy(cur, v: dict):
    vac_id = int(v["id"])
    salary = v.get("salary") or {}
    employer = v.get("employer") or {}
    area = v.get("area") or {}

    key_skills = [s.get("name") for s in (v.get("key_skills") or []) if s.get("name")]

    cur.execute(
        """
        INSERT INTO vacancies (
          vacancy_id, url, name, employer_name, area_name, published_at,
          salary_from, salary_to, salary_currency, experience, employment, schedule,
          description, key_skills, raw, updated_at
        )
        VALUES (
          %(vacancy_id)s, %(url)s, %(name)s, %(employer_name)s, %(area_name)s, %(published_at)s,
          %(salary_from)s, %(salary_to)s, %(salary_currency)s, %(experience)s, %(employment)s, %(schedule)s,
          %(description)s, %(key_skills)s, %(raw)s, now()
        )
        ON CONFLICT (vacancy_id) DO UPDATE SET
          url=EXCLUDED.url,
          name=EXCLUDED.name,
          employer_name=EXCLUDED.employer_name,
          area_name=EXCLUDED.area_name,
          published_at=EXCLUDED.published_at,
          salary_from=EXCLUDED.salary_from,
          salary_to=EXCLUDED.salary_to,
          salary_currency=EXCLUDED.salary_currency,
          experience=EXCLUDED.experience,
          employment=EXCLUDED.employment,
          schedule=EXCLUDED.schedule,
          description=EXCLUDED.description,
          key_skills=EXCLUDED.key_skills,
          raw=EXCLUDED.raw,
          updated_at=now()
        """,
        {
            "vacancy_id": vac_id,
            "url": v.get("alternate_url") or v.get("url"),
            "name": v.get("name"),
            "employer_name": employer.get("name"),
            "area_name": area.get("name"),
            "published_at": v.get("published_at"),
            "salary_from": salary.get("from"),
            "salary_to": salary.get("to"),
            "salary_currency": salary.get("currency"),
            "experience": (v.get("experience") or {}).get("name"),
            "employment": (v.get("employment") or {}).get("name"),
            "schedule": (v.get("schedule") or {}).get("name"),
            "description": v.get("description"),
            "key_skills": key_skills if key_skills else None,
            "raw": json.dumps(v, ensure_ascii=False),
        },
    )

def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

    dsn = os.environ["DB_DSN"]
    ua = os.environ["HH_USER_AGENT"]
    headers = {"User-Agent": ua}

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--area", type=int, default=113)        # 113 часто используют для РФ
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--per_page", type=int, default=100)    # max 100
    args = ap.parse_args()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for page in range(args.pages):
                params = {
                    "text": args.query,
                    "area": args.area,
                    "page": page,
                    "per_page": args.per_page,
                }
                data = fetch_json(f"{API}/vacancies", headers=headers, params=params)

                cur.execute(
                    "INSERT INTO vacancies_raw(query_text, area, page, response) VALUES (%s, %s, %s, %s::jsonb)",
                    (args.query, args.area, page, json.dumps(data, ensure_ascii=False)),
                )

                items = data.get("items", [])
                ids = [str(it["id"]) for it in items if "id" in it]

                for vac_id in tqdm(ids, desc=f"page {page} details", leave=False):
                    v = fetch_json(f"{API}/vacancies/{vac_id}", headers=headers)
                    upsert_vacancy(cur, v)

            conn.commit()

    print("Done.")

if __name__ == "__main__":
    main()
