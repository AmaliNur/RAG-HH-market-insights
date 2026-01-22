import json
import requests
import streamlit as st

API_BASE_DEFAULT = "http://127.0.0.1:8000"


# -----------------------------
# Helpers
# -----------------------------
def api_get(base: str, path: str, params: dict | None = None, timeout: int = 60):
    url = base.rstrip("/") + path
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def format_badge(text: str):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:0.15rem 0.55rem;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.2);
            font-size:0.85rem;
            margin-right:0.35rem;
            margin-bottom:0.25rem;">
            {text}
        </span>
        """,
        unsafe_allow_html=True
    )


def show_evidence(evidence: list, max_items: int = 2):
    if not evidence:
        st.caption("Нет evidence-цитат.")
        return
    for ev in evidence[:max_items]:
        meta = f"chunk {ev.get('chunk_no')} | score={ev.get('score'):.4f} dist={ev.get('dist'):.4f} kw={ev.get('kw_sim'):.4f}"
        st.caption(meta)
        st.write(ev.get("text", ""))


def show_why(why: list):
    if not why:
        return
    for block in why:
        t = block.get("type")
        items = block.get("items") or []
        if not items:
            continue
        title = "Почему подходит" if t in ("query_match", "tech_terms") else f"{t}"
        st.caption(title + ": " + ", ".join(items))


def show_vacancy_card(v: dict, evidence_max: int = 2, show_debug: bool = False):
    title = v.get("name") or "Без названия"
    company = v.get("employer_name") or "—"
    area = v.get("area_name") or "—"
    url = v.get("url") or ""

    cols = st.columns([0.78, 0.22])
    with cols[0]:
        st.subheader(title)
        st.write(f"**{company}** · {area}")
        if url:
            st.link_button("Открыть вакансию", url, use_container_width=False)
    with cols[1]:
        st.metric("score", f"{v.get('best_score', 0):.4f}")

    if v.get("why"):
        show_why(v["why"])

    st.markdown("**Evidence (цитаты):**")
    show_evidence(v.get("evidence", []), max_items=evidence_max)

    if show_debug:
        with st.expander("Raw JSON"):
            st.code(json.dumps(v, ensure_ascii=False, indent=2), language="json")


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="HH RAG: вакансии + аналитика",
    layout="wide",
)

st.title("HH RAG: поиск вакансий и аналитика рынка (RAG + pgvector)")

with st.sidebar:
    st.header("Настройки")
    api_base = st.text_input("FastAPI base URL", value=API_BASE_DEFAULT)
    st.divider()

    st.subheader("Параметры retrieval")
    k = st.slider("k (вакансий)", 3, 20, 8)
    per_vac = st.slider("цитат на вакансию", 1, 4, 2)
    candidates = st.slider("candidates (топ чанков для перескоринга)", 50, 800, 250, step=50)
    kw_weight = st.slider("kw_weight (hybrid)", 0.0, 0.8, 0.25, step=0.05)
    max_quote = st.slider("макс длина цитаты", 200, 1500, 700, step=50)
    do_highlight = st.checkbox("подсвечивать ключевые слова", value=True)
    show_debug = st.checkbox("показывать Raw JSON", value=False)

    st.divider()
    st.subheader("Аналитика")
    market_limit = st.slider("limit для market", 5, 50, 15)

# Top status bar
status_cols = st.columns([0.25, 0.25, 0.25, 0.25])
try:
    health = api_get(api_base, "/health")
    stats = api_get(api_base, "/stats")
    with status_cols[0]:
        st.success("API: OK" if health.get("status") == "ok" else "API: ?")
    with status_cols[1]:
        st.metric("vacancies", stats.get("vacancies_total", 0))
    with status_cols[2]:
        st.metric("with description", stats.get("vacancies_with_description", 0))
    with status_cols[3]:
        emb = stats.get("chunks_with_embedding", 0)
        ch = stats.get("chunks_total", 0)
        st.metric("chunks embedded", f"{emb}/{ch}")
except Exception as e:
    st.error(f"Не удалось подключиться к API: {e}")
    st.stop()

# Tabs
tab_search, tab_ask, tab_market, tab_about = st.tabs(["Search", "Ask (RAG)", "Market", "About"])

with tab_search:
    st.subheader("Поиск вакансий (vacancy-level retrieval)")
    q = st.text_input("Запрос", value="data engineer airflow kafka", key="search_query")
    run = st.button("Search", type="primary")

    if run:
        try:
            resp = api_get(
                api_base,
                "/search",
                params={
                    "q": q,
                    "k": k,
                    "per_vac": per_vac,
                    "candidates": candidates,
                    "kw_weight": kw_weight,
                    "max_quote": max_quote,
                    "do_highlight": do_highlight,
                },
            )
            results = resp.get("results", [])
            st.caption(f"Hybrid used: {resp.get('hybrid_used')} | kw_weight_used={resp.get('kw_weight_used')}")
            st.write(f"Найдено вакансий: **{len(results)}**")

            for v in results:
                st.divider()
                show_vacancy_card(v, evidence_max=per_vac, show_debug=show_debug)

        except Exception as e:
            st.error(f"Ошибка /search: {e}")

with tab_ask:
    st.subheader("Ask (RAG): ответ + источники")
    q2 = st.text_input("Запрос", value="стажировка backend java spring москва", key="ask_query")
    run2 = st.button("Ask", type="primary")

    if run2:
        try:
            resp = api_get(
                api_base,
                "/ask",
                params={
                    "q": q2,
                    "k": min(k, 12),
                    "per_vac": per_vac,
                    "candidates": candidates,
                    "kw_weight": kw_weight,
                    "max_quote": max_quote,
                    "do_highlight": do_highlight,
                },
            )

            summary = resp.get("summary", {})
            st.caption(f"Hybrid used: {resp.get('hybrid_used')} | kw_weight_used={resp.get('kw_weight_used')}")
            if summary:
                st.markdown("### Summary")
                st.write(summary.get("text", ""))

                if summary.get("tech_signals"):
                    st.markdown("**Tech signals:**")
                    for t in summary["tech_signals"]:
                        format_badge(t)

                if summary.get("query_keywords"):
                    st.markdown("**Query keywords:**")
                    for t in summary["query_keywords"]:
                        format_badge(t)

                if summary.get("notes"):
                    with st.expander("Notes"):
                        for n in summary["notes"]:
                            st.write("• " + n)

            results = resp.get("results", [])
            st.markdown("### Results")
            st.write(f"Найдено вакансий: **{len(results)}**")

            for v in results:
                st.divider()
                show_vacancy_card(v, evidence_max=per_vac, show_debug=show_debug)

        except Exception as e:
            st.error(f"Ошибка /ask: {e}")

with tab_market:
    st.subheader("Аналитика рынка (по корпусу)")
    col1, col2, col3 = st.columns(3)

    try:
        geo = api_get(api_base, "/market/geo", params={"limit": market_limit})
        emp = api_get(api_base, "/market/employers", params={"limit": market_limit})
        tech = api_get(api_base, "/market/tech-top", params={"limit": market_limit})

        with col1:
            st.markdown("### Top geo")
            for row in geo.get("top", []):
                st.write(f"{row['area_name']}: **{row['count']}**")

        with col2:
            st.markdown("### Top employers")
            for row in emp.get("top", []):
                st.write(f"{row['employer_name']}: **{row['count']}**")

        with col3:
            st.markdown("### Top tech terms")
            for row in tech.get("top", []):
                st.write(f"{row['term']}: **{row['count']}**")

    except Exception as e:
        st.error(f"Ошибка market endpoints: {e}")
