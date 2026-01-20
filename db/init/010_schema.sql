CREATE TABLE IF NOT EXISTS vacancies_raw (
  id            BIGSERIAL PRIMARY KEY,
  fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  query_text    TEXT NOT NULL,
  area          INT,
  page          INT,
  response      JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS vacancies (
  vacancy_id      BIGINT PRIMARY KEY,
  url             TEXT,
  name            TEXT,
  employer_name   TEXT,
  area_name       TEXT,
  published_at    TIMESTAMPTZ,
  salary_from     INT,
  salary_to       INT,
  salary_currency TEXT,
  experience      TEXT,
  employment      TEXT,
  schedule        TEXT,
  description     TEXT,
  key_skills      TEXT[],
  raw             JSONB NOT NULL,
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
