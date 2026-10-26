# 🩺 MediGuide AI

AI-powered medical symptom assessment and patient guidance assistant built with **LangChain + OpenAI + Streamlit**.

> ⚠️ **Medical disclaimer:** MediGuide AI is an educational prototype. It is NOT a doctor, never gives a confirmed diagnosis, and is not a replacement for professional medical advice, emergency services or treatment. In an emergency call your local emergency number immediately.

## Features

**Required by the assignment**
- Form with age, gender, symptoms (multiselect + free text), duration, severity slider, conditions, medications, notes and answer language
- `ChatOpenAI`, `PromptTemplate`, `ChatPromptTemplate`, `SystemMessage` / `HumanMessage` / `AIMessage`, reusable `LLMChain`
- Strict JSON output, parsed safely (fences stripped, trailing commas fixed, never crashes)
- Live streaming with `llm.stream()` + `st.write_stream()`
- Both `InMemoryCache` and `SQLiteCache`, switchable in the sidebar
- Urgency colours (LOW / MEDIUM / HIGH / EMERGENCY) and disclaimers in the sidebar, main area, results and a fixed footer

**Extras**
- Animated theme engine: dark / light mode, 4 colour palettes, reveal animation on theme change, reduce-motion toggle
- API-key screen shown first when no key is found in `.env` (key is validated before use)
- Local red-flag safety pre-screen and self-harm crisis message (runs before the model)
- One-click sample test scenarios
- Session history with export, analytics dashboard (urgency distribution, top symptoms, severity timeline)
- PDF report (Latin-script languages), HTML report (all languages, RTL for Urdu/Arabic) and raw JSON download
- 11 answer languages, 4 model choices, "LangChain Lab" tab with a cache benchmark

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env
# edit .env and paste your key: OPENAI_API_KEY=sk-...
streamlit run app.py
```

If you skip the `.env` step the app asks for your key on its first screen. **Never commit `.env`.**

## Project structure

```
medical_ai_assistant/
├── app.py              Streamlit UI (run this)
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
├── src/
│   ├── config.py         settings, form options, urgency styles, palettes, red-flag rules
│   ├── prompts.py        PromptTemplate + ChatPromptTemplate + JSON schema
│   ├── chains.py         ChatOpenAI, LLMChain, streaming, message demo, cache benchmark
│   ├── cache_manager.py  in-memory + SQLite caching
│   └── utils.py          safe JSON parsing, safety pre-screen, PDF/HTML reports
└── docs/Medical_AI_Assignment.pdf
```

Data flow: form input → prompt inputs → select cache → ChatOpenAI + LLMChain → run chain → parse JSON → stream narrative → dashboard.

## Caching: InMemoryCache vs SQLiteCache

| | InMemoryCache | SQLiteCache |
|---|---|---|
| Stored in | RAM | A file on disk (`.cache/langchain_cache.db`) |
| Speed | Fastest | Fast, slightly slower |
| Survives restart? | No | Yes |
| Best for | One session | Reusing results across sessions |

The cache is registered once with `set_llm_cache(...)` and LangChain checks it before every model call. Submit the same form twice with caching on: the second run returns in a fraction of a second (the Cache metric shows ⚡ HIT). The LLM cache covers the JSON assessment; the streamed narrative is replayed from a small in-process memo because streaming does not always pass through the LLM cache.

## Testing

Use the sample buttons or the table in the "About & Safety" tab: cold (LOW), fever + cough (MEDIUM/HIGH), chest pain + breathlessness (HIGH/EMERGENCY), same form twice (cache), empty symptoms (warning, no API call), Urdu output.

## Limitations

Educational prototype only. Model output can be wrong or incomplete and must never be used for real diagnosis or treatment.
