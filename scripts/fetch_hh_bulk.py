import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

API = "https://api.hh.ru"

DEFAULT_QUERIES = [
    "backend",
    "java spring",
    "python developer",
    "frontend react",
    "devops",
    "qa engineer",
    "data engineer",
    "data analyst",
    "data scientist",
    "mlops",
    "golang",
    "c++",
    "android developer",
    "ios developer",
]

DEFAULT_AREAS = [
    1, 2, 3, 4,  # Москва, СПб, Екб, Новосиб
    66, 78, 88,  # НН, Самара, Казань
    113          # Россия (широкий)
]

class CaptchaRequired(Exception):
    def __init__(self, captcha_url: str):
        super().__init__("captcha_required")
        self.captcha_url = captcha_url

def fetch_json(session: requests.Session, url: str, headers: dict, params: Optional[dict] = None, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=headers, params=params, timeout=45)

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "2"))
                print(f"[429] rate limited, sleep {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 403:
                try:
                    j = r.json()
                except Exception:
                    j = None
                if isinstance(j, dict) and "errors" in j:
                    for e in j.get("errors", []):
                        if e.get("value") == "captcha_required":
                            raise CaptchaRequired(e.get("captcha_url", ""))
                r.raise_for_status()

            if r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue

            r.raise_for_status()
            return r.json()
        except CaptchaRequired:
            raise
        except Exception as e:
            last = e
            time.sleep(1.2 * (attempt + 1))
    raise last if last else RuntimeError("fetch failed")

def upsert_from_list_item(cur, it: dict) -> str:
    """
    Upsert по vacancy_id из LIST endpoint (/vacancies).
    Возвращает: "inserted" или "updated"
    """
    vac_id = int(it["id"])
    employer = it.get("employer") or {}
    area = it.get("area") or {}
    salary = it.get("salary") or {}

    # existed?
    cur.execute("select 1 from vacancies where vacancy_id=%s", (vac_id,))
    exists = cur.fetchone() is not None

    cur.execute(
        """
        insert into vacancies(
          vacancy_id, url, name, employer_name, area_name, published_at,
          salary_from, salary_to, salary_currency,
          experience, employment, schedule,
          description, key_skills, raw, updated_at
        )
        values(
          %s,%s,%s,%s,%s,%s,
          %s,%s,%s,
          %s,%s,%s,
          null, null, %s::jsonb, now()
        )
        on conflict (vacancy_id) do update set
          url=excluded.url,
          name=excluded.name,
          employer_name=excluded.employer_name,
          area_name=excluded.area_name,
          published_at=excluded.published_at,
          salary_from=excluded.salary_from,
          salary_to=excluded.salary_to,
          salary_currency=excluded.salary_currency,
          experience=excluded.experience,
          employment=excluded.employment,
          schedule=excluded.schedule,
          raw=excluded.raw,
          updated_at=now()
        """,
        (
            vac_id,
            it.get("alternate_url") or it.get("url"),
            it.get("name"),
            employer.get("name"),
            area.get("name"),
            it.get("published_at"),
            salary.get("from"),
            salary.get("to"),
            salary.get("currency"),
            (it.get("experience") or {}).get("name"),
            (it.get("employment") or {}).get("name"),
            (it.get("schedule") or {}).get("name"),
            json.dumps(it, ensure_ascii=False),
        ),
    )
    return "updated" if exists else "inserted"

def insert_raw_page(cur, query_text: str, area: int, page: int, data: dict):
    cur.execute(
        "insert into vacancies_raw(query_text, area, page, response) values (%s,%s,%s,%s::jsonb)",
        (query_text, area, page, json.dumps(data, ensure_ascii=False)),
    )

def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    dsn = os.environ["DB_DSN"]
    ua = os.environ["HH_USER_AGENT"]
    headers = {"User-Agent": ua, "Accept": "application/json"}

    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-per-pair", type=int, default=2)
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--list-delay", type=float, default=0.2)
    ap.add_argument("--list-jitter", type=float, default=0.15)
    args = ap.parse_args()

    queries = DEFAULT_QUERIES
    areas = DEFAULT_AREAS

    pair_total = len(queries) * len(areas)
    pair_idx = 0

    session = requests.Session()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from vacancies")
            start_cnt = cur.fetchone()[0]
            print("DB starting vacancies count:", start_cnt)

        for q in queries:
            for area in areas:
                pair_idx += 1
                print(f"\n=== [{pair_idx}/{pair_total}] query='{q}' area={area} ===")
                pair_ins = 0
                pair_upd = 0
                pair_err = 0
                t0 = time.time()

                for page in range(args.pages_per_pair):
                    # небольшой rate limit на LIST
                    time.sleep(args.list_delay + random.uniform(0, args.list_jitter))

                    params = {"text": q, "area": area, "page": page, "per_page": args.per_page}
                    data = fetch_json(session, f"{API}/vacancies", headers=headers, params=params)
                    items = data.get("items") or []
                    print(f"  page={page} found={data.get('found')} page_items={len(items)}")

                    with conn.cursor() as cur:
                        insert_raw_page(cur, q, area, page, data)

                        for it in tqdm(items, desc=f"  upsert p{page}", leave=False):
                            try:
                                res = upsert_from_list_item(cur, it)
                                if res == "inserted":
                                    pair_ins += 1
                                else:
                                    pair_upd += 1
                            except Exception as e:
                                pair_err += 1
                                print("  ! upsert error:", e)

                        conn.commit()

                    if not items:
                        break

                with conn.cursor() as cur:
                    cur.execute("select count(*) from vacancies")
                    total = cur.fetchone()[0]

                print(f"=== DONE pair in {time.time()-t0:.1f}s | inserted={pair_ins} updated={pair_upd} errors={pair_err} | db_total={total} ===")

    session.close()

if __name__ == "__main__":
    main()
