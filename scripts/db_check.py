import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

dsn = os.environ["DB_DSN"]

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("select 1;")
        print("DB OK:", cur.fetchone())
        cur.execute("select tablename from pg_tables where schemaname='public' order by tablename;")
        print("Tables:", [r[0] for r in cur.fetchall()])
