import os
import json
import time
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
import psycopg
from dotenv import load_dotenv

API = "https://api.hh.ru"


class CaptchaRequired(Exception):
    def __init__(self, captcha_url: str, request_id: Optional[str] = None):
        super().__init__("captcha_required")
        self.captcha_url = captcha_url
        self.request_id = request_id


def fetch_json_with_retries(
    session: requests.Session,
    url: str,
    headers: dict,
    timeout: int = 45,
    retries: int = 4,
    backoff_base: float = 1.7,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Возвращает (json, status_tag)
      - (dict, None) если успешно
      - (None, "not_found") если 404
      - (None, "rate_limited") если 429 после ретраев
      - (None, "http_error") для прочих неуспехов после ретраев

    CAPTCHA -> бросает CaptchaRequired
    """
    last_exc = None

    for attempt in range(retries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout)

            # 404: вакансию могли удалить/скрыть
            if r.status_code == 404:
                return None, "not_found"

            # 429: слишком много запросов
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                sleep_s = float(ra) if ra and ra.replace(".", "").isdigit() else (backoff_base ** attempt)
                time.sleep(min(60.0, sleep_s))
                continue

            # 403: проверяем captcha_required
            if r.status_code == 403:
                try:
                    j = r.json()
                except Exception:
                    j = None
                if isinstance(j, dict) and "errors" in j:
                    for e in j.get("errors", []):
                        if e.get("value") == "captcha_required":
                            raise CaptchaRequired(
                                captcha_url=e.get("captcha_url", ""),
                                request_id=j.get("request_id"),
                            )
                r.raise_for_status()

            # 5xx: временно, ретраим
            if r.status_code >= 500:
                time.sleep(min(30.0, backoff_base ** attempt))
                continue

            r.raise_for_status()
            return r.json(), None

        except CaptchaRequired:
            raise
        except Exception as e:
            last_exc = e
            time.sleep(min(30.0, backoff_base ** attempt))

    # после всех попыток
    return None, "rate_limited" if isinstance(last_exc, requests.HTTPError) and getattr(last_exc.response, "status_code", None) == 429 else "http_error"


def extract_key_skills(detail: Dict[str, Any]) -> List[str]:
    ks = detail.get("key_skills") or []
    out: List[str] = []
    for item in ks:
        n = (item or {}).get("name")
        if n:
            out.append(n)
    return out


def detect_column_info(cur, table: str, column: str) -> Optional[Tuple[str, str]]:
    cur.execute(
        """
        select data_type, udt_name
        from information_schema.columns
        where table_schema='public' and table_name=%s and column_name=%s
        """,
        (table, column),
    )
    row = cur.fetchone()
    if not row:
        return None
    return row[0], row[1]


def detect_columns(cur, table: str) -> set:
    cur.execute(
        """
        select column_name
        from information_schema.columns
        where table_schema='public' and table_name=%s
        """,
        (table,),
    )
    return {r[0] for r in cur.fetchall()}


def detect_key_skills_mode(cur) -> str:
    """
    Возвращает: "jsonb" | "text" | "text_array" | "unknown"
    """
    info = detect_column_info(cur, "vacancies", "key_skills")
    if not info:
        return "unknown"

    data_type, udt_name = info

    # jsonb
    if data_type == "jsonb" or udt_name == "jsonb":
        return "jsonb"

    # простой text
    if data_type == "text":
        return "text"

    # массив (чаще всего text[])
    # в information_schema для массивов data_type='ARRAY', udt_name типа '_text'
    if data_type == "ARRAY" and udt_name in ("_text", "_varchar", "_bpchar"):
        return "text_array"

    return "unknown"


def select_ids_to_enrich(cur, limit: int, order: str) -> List[int]:
    order_sql = "published_at desc nulls last"
    if order == "oldest":
        order_sql = "published_at asc nulls last"
    elif order == "random":
        order_sql = "random()"

    cur.execute(
        f"""
        select vacancy_id
        from vacancies
        where description is null or description = ''
        order by {order_sql}
        limit %s
        """,
        (limit,),
    )
    return [int(r[0]) for r in cur.fetchall()]


def main():
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    ua = os.getenv("HH_USER_AGENT")
    if not dsn or not ua:
        raise RuntimeError("Нужны DB_DSN и HH_USER_AGENT в .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="Сколько вакансий обогатить за запуск")
    ap.add_argument("--order", choices=["newest", "oldest", "random"], default="newest", help="Какие обогащать первыми")
    ap.add_argument("--delay", type=float, default=2.5, help="Базовая задержка между запросами (сек)")
    ap.add_argument("--jitter", type=float, default=1.0, help="Случайная добавка к задержке (сек)")
    ap.add_argument("--timeout", type=int, default=45, help="Timeout HTTP запроса (сек)")
    ap.add_argument("--retries", type=int, default=4, help="Ретраи на сетевые/5xx/429")
    ap.add_argument("--commit-every", type=int, default=10, help="Коммитить раз в N обновлений (ускоряет)")
    args = ap.parse_args()

    headers = {"User-Agent": ua, "Accept": "application/json"}
    session = requests.Session()

    with psycopg.connect(dsn) as conn:
        conn.autocommit = False

        with conn.cursor() as cur:
            cols = detect_columns(cur, "vacancies")
            ks_mode = detect_key_skills_mode(cur)

            print("Detected vacancies columns:", ", ".join(sorted(list(cols))[:12]) + (" ..." if len(cols) > 12 else ""))
            print("key_skills mode:", ks_mode)

            ids = select_ids_to_enrich(cur, limit=args.limit, order=args.order)

        print("to enrich:", len(ids))

        updated = 0
        not_found = 0
        rate_limited = 0
        errors = 0

        pending_updates = 0

        try:
            for idx, vac_id in enumerate(ids, 1):
                time.sleep(args.delay + random.uniform(0, args.jitter))

                detail, tag = fetch_json_with_retries(
                    session,
                    f"{API}/vacancies/{vac_id}",
                    headers=headers,
                    timeout=args.timeout,
                    retries=args.retries,
                )

                if tag == "not_found":
                    not_found += 1
                    continue
                if tag == "rate_limited":
                    rate_limited += 1
                    continue
                if tag == "http_error":
                    errors += 1
                    continue
                if not detail:
                    errors += 1
                    continue

                desc = detail.get("description") or ""
                ks = extract_key_skills(detail)

                with conn.cursor() as cur:
                    set_parts = []
                    values = []

                    if "description" in cols:
                        set_parts.append("description=%s")
                        values.append(desc)

                    if "key_skills" in cols and ks_mode != "unknown":
                        if ks_mode == "jsonb":
                            set_parts.append("key_skills=%s::jsonb")
                            values.append(json.dumps(ks, ensure_ascii=False))
                        elif ks_mode == "text":
                            set_parts.append("key_skills=%s")
                            values.append(", ".join(ks))
                        elif ks_mode == "text_array":
                            # psycopg умеет адаптировать list[str] в text[]
                            set_parts.append("key_skills=%s")
                            values.append(ks)

                    # бонус: если есть колонка под raw details — сохраним
                    if "raw_details" in cols:
                        set_parts.append("raw_details=%s::jsonb")
                        values.append(json.dumps(detail, ensure_ascii=False))
                    elif "details" in cols:
                        set_parts.append("details=%s::jsonb")
                        values.append(json.dumps(detail, ensure_ascii=False))
                    elif "payload" in cols:
                        set_parts.append("payload=%s::jsonb")
                        values.append(json.dumps(detail, ensure_ascii=False))

                    if "updated_at" in cols:
                        set_parts.append("updated_at=now()")

                    if not set_parts:
                        # нечего обновлять
                        continue

                    values.append(vac_id)
                    sql = f"update vacancies set {', '.join(set_parts)} where vacancy_id=%s"
                    cur.execute(sql, values)

                updated += 1
                pending_updates += 1

                if updated % 25 == 0:
                    print(f"enriched: {updated} | not_found={not_found} rate_limited={rate_limited} errors={errors}")

                if pending_updates >= args.commit_every:
                    conn.commit()
                    pending_updates = 0

            if pending_updates > 0:
                conn.commit()

        except CaptchaRequired as ce:
            # фиксируем уже накопленные изменения
            try:
                conn.commit()
            except Exception:
                pass

            print("\n!!! CAPTCHA REQUIRED !!!")
            print("captcha_url:", ce.captcha_url)
            if ce.request_id:
                print("request_id:", ce.request_id)
            print("Остановились. Увеличь delay/jitter и запускай позже.")
            return

    print(f"DONE: enriched={updated} not_found={not_found} rate_limited={rate_limited} errors={errors}")


if __name__ == "__main__":
    main()
