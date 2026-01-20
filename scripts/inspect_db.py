import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
dsn = os.environ["DB_DSN"]

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("select count(*) from vacancies;")
        print("vacancies:", cur.fetchone()[0])

        cur.execute("select count(*) from vacancies_raw;")
        print("vacancies_raw:", cur.fetchone()[0])

        cur.execute("""
            select vacancy_id, name, employer_name, area_name, published_at
            from vacancies
            order by published_at desc nulls last
            limit 5;
        """)
        for row in cur.fetchall():
            print(row)
