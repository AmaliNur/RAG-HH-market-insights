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

CREATE TABLE IF NOT EXISTS vacancy_chunks (
  id BIGSERIAL PRIMARY KEY,
  vacancy_id BIGINT NOT NULL REFERENCES vacancies(vacancy_id) ON DELETE CASCADE,
  chunk_no INT NOT NULL,
  text TEXT NOT NULL,
  source_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding vector(384)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_vacancy_chunks ON vacancy_chunks(vacancy_id, chunk_no);
CREATE INDEX IF NOT EXISTS idx_vacancy_chunks_vacancy_id ON vacancy_chunks(vacancy_id);