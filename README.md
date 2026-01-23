# 🔍 HH RAG Project — Semantic Search & Market Analytics

> RAG-система для семантического поиска и анализа IT-вакансий РФ на данных HeadHunter API  
> **Включает:** сбор корпуса → enrichment деталей → chunking → embeddings (pgvector) → hybrid retrieval → FastAPI + Streamlit UI

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-Embeddings-green?style=flat)](https://github.com/pgvector/pgvector)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📌 Содержание

- [О проекте](#-о-проекте)
- [Возможности](#-возможности)
- [Технологический стек](#-технологический-стек)
- [Архитектура](#-архитектура)
- [Быстрый старт](#-быстрый-старт)
- [Пайплайн формирования корпуса](#-пайплайн-формирования-корпуса)
- [Запуск API и UI](#-запуск-api-и-ui)
- [Использование API](#-использование-api)
- [Решение проблем](#-решение-проблем)
- [Управление Docker](#-управление-docker)
- [Структура проекта](#-структура-проекта)

---

## 🎯 О проекте

Проект реализует **семантический поиск вакансий** (RAG retrieval с evidence-цитатами) и **аналитику рынка** на базе корпуса IT-вакансий из HeadHunter.

### Ключевые особенности

- 🧠 **Семантический поиск** — поиск по смыслу через векторные эмбеддинги
- 🎯 **Vacancy-level ranking** — выдача на уровне вакансий, а не списком чанков (меньше дублей)
- 📋 **Evidence-цитаты** — объяснение релевантности результатов
- 📊 **Аналитика рынка** — география, работодатели, технологические тренды
- 🚀 **FastAPI + Streamlit** — удобный REST API и интерактивный UI

---

## ✨ Возможности

### 📦 Корпус данных

| Функция | Описание |
|---------|----------|
| **Bulk ingestion** | Массовая загрузка вакансий из HH API по запросам и регионам |
| **Enrichment** | Детализация вакансий (description, key_skills) |
| **Raw storage** | Сохранение оригинальных ответов для воспроизводимости |

### 🔎 RAG Retrieval

| Компонент | Технология |
|-----------|------------|
| **Chunking** | Нарезка текста вакансий на фрагменты |
| **Embeddings** | FastEmbed → хранение в pgvector |
| **Hybrid scoring** | Vector similarity + keyword similarity (pg_trgm) |
| **Ranking** | Vacancy-level с evidence-цитатами |

### 🌐 Serving

**FastAPI эндпоинты:**
- `/health`, `/stats` — статус системы
- `/search` — поиск вакансий с evidence
- `/ask` — RAG-summary с результатами и обоснованием
- `/market/*` — аналитика (география / компании / технологии)

**Streamlit UI:**
- Вкладки: Search / Ask / Market
- Интерактивная визуализация результатов

---

## 🧰 Технологический стек

- **Python 3.x** — основной язык разработки
- **PostgreSQL 16** — база данных
- **pgvector** — векторные эмбеддинги
- **pg_trgm** — текстовое сходство для keyword similarity
- **FastEmbed** — локальная генерация эмбеддингов
- **FastAPI** — REST API backend
- **Streamlit** — интерактивный UI
- **Docker Compose** — контейнеризация БД

---

## 🏗️ Архитектура

```
HH API
  └─> fetch_hh_bulk.py → vacancies / vacancies_raw
      └─> enrich_hh_details.py → description, key_skills
          └─> build_chunks.py → vacancy_chunks (text chunks)
              └─> embed_chunks.py → vacancy_chunks.embedding (pgvector)
                  └─> FastAPI (/search, /ask, /market/*)
                      └─> Streamlit UI (Search / Ask / Market)
```

---

## 🚀 Быстрый старт

### 1️⃣ Клонирование репозитория

```bash
git clone <REPO_URL>
cd RAG_project
```

### 2️⃣ Настройка Python окружения

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Если PowerShell блокирует активацию:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
HH_USER_AGENT=hh-rag-project/0.1 (your_email@example.com)
DB_DSN=host=localhost port=5433 dbname=rag user=postgres password=postgres
```

> ⚠️ **Важно:**
> - `HH_USER_AGENT` обязателен — HH может чаще требовать captcha без корректного User-Agent
> - Порт в DSN должен быть `5433` (как в `docker-compose.yml`)

### 4️⃣ Запуск базы данных

```bash
docker compose up -d
docker compose ps
```

**Проверка подключения:**

```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -c "SELECT 1;"
```

### 5️⃣ Проверка инициализации БД

**Проверьте наличие таблиц:**

```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -c "\dt"
```

**Если таблица `vacancy_chunks` отсутствует, примените схему вручную:**

Вариант A (SQL внутри контейнера):
```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -f /docker-entrypoint-initdb.d/010_schema.sql
```

Вариант B (SQL в репозитории):
```bash
docker exec -i rag_project-db-1 psql -U postgres -d rag < 010_schema.sql
```

---

## 🧱 Пайплайн формирования корпуса

### Шаг 1: Bulk-загрузка вакансий

Загрузите корпус вакансий с HH API:

```bash
python scripts/fetch_hh_bulk.py --pages-per-pair 2 --per-page 100
```

**Проверка количества вакансий:**

```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -c "SELECT COUNT(*) FROM vacancies;"
```

### Шаг 2: Enrichment деталей

Обогатите вакансии подробной информацией (description, key_skills):

```bash
python scripts/enrich_hh_details.py --limit 200 --delay 2.5 --jitter 1.0
```

> 💡 **Совет:** Запускайте enrichment постепенно — HH может отдавать `403 captcha_required` при частых запросах.

**Проверка заполненности:**

```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -c "SELECT COUNT(*) FROM vacancies WHERE description IS NOT NULL AND description <> '';"
```

### Шаг 3: Chunking текста

Разбейте тексты вакансий на чанки:

```bash
python scripts/build_chunks.py --rebuild 1
```

**Проверка чанков:**

```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -c "SELECT COUNT(*) AS chunks, COUNT(DISTINCT vacancy_id) AS covered FROM vacancy_chunks;"
```

### Шаг 4: Генерация эмбеддингов

Вычислите векторные представления:

```bash
python scripts/embed_chunks.py --batch 64
```

**Проверка эмбеддингов:**

```bash
docker exec -it rag_project-db-1 psql -U postgres -d rag -c "SELECT COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS embedded, COUNT(*) AS total FROM vacancy_chunks;"
```

---

## ▶️ Запуск API и UI

### FastAPI Backend

Запустите REST API:

```bash
uvicorn api.main:app --reload --port 8000
```

**Swagger UI (документация API):**

```
http://127.0.0.1:8000/docs
```

### Streamlit UI

Запустите интерактивный интерфейс:

```bash
streamlit run ui/app.py
```

**Web UI:**

```
http://localhost:8501
```

---

## 🧪 Использование API

### Health & Stats

```bash
# Проверка состояния системы
curl http://127.0.0.1:8000/health

# Статистика корпуса
curl http://127.0.0.1:8000/stats
```

### Search (поиск вакансий)

```bash
curl "http://127.0.0.1:8000/search?q=data%20engineer%20airflow%20kafka&k=8&per_vac=2&candidates=250&kw_weight=0.25"
```

**Параметры:**
- `q` — поисковый запрос
- `k` — количество результатов
- `per_vac` — чанков на вакансию
- `candidates` — количество кандидатов для ранжирования
- `kw_weight` — вес keyword similarity (0.0-1.0)

### Ask (RAG summary с evidence)

```bash
curl "http://127.0.0.1:8000/ask?q=backend%20java%20spring%20стажировка%20москва&k=8&per_vac=2&candidates=250&kw_weight=0.25"
```

### Market Analytics

```bash
# География вакансий
curl "http://127.0.0.1:8000/market/geo?limit=15"

# Топ работодателей
curl "http://127.0.0.1:8000/market/employers?limit=15"

# Популярные технологии
curl "http://127.0.0.1:8000/market/tech-top?limit=20"
```

---

## 🔧 Решение проблем

### Проблема: HH отдает `403 captcha_required`

Это нормально при частых запросах к API.

**Решения:**
- ⏱️ Увеличьте `--delay` и `--jitter` в enrichment
- 📦 Делайте enrichment пакетами (по 200-500 вакансий)
- 📧 Убедитесь, что `HH_USER_AGENT` содержит реальную почту
- ⏸️ При необходимости остановитесь и продолжите позже

### Проблема: Таблицы не создались

Init SQL выполняется только на чистом volume.

**Решение (пересоздание):**

```bash
docker compose down -v  # ⚠️ Удалит данные
docker compose up -d
```

**Решение (без удаления данных):**

См. раздел [Проверка инициализации БД](#5️⃣-проверка-инициализации-бд)

### Проблема: Порт 5433 занят

**Измените порт в `docker-compose.yml`:**

```yaml
ports:
  - "5434:5432"  # Вместо 5433:5432
```

**Обновите `.env`:**

```env
DB_DSN=host=localhost port=5434 dbname=rag user=postgres password=postgres
```

---

## 🐳 Управление Docker

### Остановка контейнеров (данные сохраняются)

```bash
docker compose down
```

### ⚠️ Полная очистка (удаление всех данных)

```bash
docker compose down -v
```

> **Внимание:** Эта команда удалит volume `db_data` со всеми данными!

---

## 📁 Структура проекта

```
RAG_project/
├── 📂 api/
│   └── main.py                    # FastAPI backend
├── 📂 ui/
│   └── app.py                     # Streamlit UI
├── 📂 scripts/
│   ├── fetch_hh_bulk.py           # Bulk ingestion (HH API)
│   ├── enrich_hh_details.py       # Enrichment: description/key_skills
│   ├── build_chunks.py            # Chunking
│   └── embed_chunks.py            # Embeddings → pgvector
├── 📂 db/
│   └── init/
│       ├── 001_extensions.sql     # Расширения PostgreSQL
│       └── 010_schema.sql         # Схема таблиц
├── docker-compose.yml             # Docker конфигурация
├── requirements.txt               # Python зависимости
├── .env.example                   # Пример переменных окружения
└── README.md
```

---

## 📊 Примеры использования

### Полный пайплайн (быстрый тест)

```bash
# 1. Поднять БД
docker compose up -d

# 2. Загрузить вакансии
python scripts/fetch_hh_bulk.py --pages-per-pair 1 --per-page 100

# 3. Обогатить данные
python scripts/enrich_hh_details.py --limit 100 --delay 2.5

# 4. Создать чанки
python scripts/build_chunks.py --rebuild 1

# 5. Сгенерировать эмбеддинги
python scripts/embed_chunks.py --batch 64

# 6. Запустить API
uvicorn api.main:app --reload --port 8000

# 7. Запустить UI (в другом терминале)
streamlit run ui/app.py
```

---

## 📝 Лицензия

Проект создан в образовательных целях.

---

## 🤝 Вклад в проект

Если у вас есть предложения или вы нашли баг, создайте Issue в репозитории.

---

**Сделано с ❤️ для анализа IT-рынка вакансий**