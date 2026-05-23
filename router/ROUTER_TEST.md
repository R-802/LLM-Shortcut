# Meta-router smoke tests

## Setup

1. Run **`setup.bat`** in the project root.
   - Prompts for Python paths, optional ntfy notifications, and API keys.
   - Writes `.env`, installs requirements, optionally registers Windows logon startup.
2. For development restarts: **`restart_service.bat`** (add `--nopause` to skip the final pause).
3. Put study files in **`context/`** (PDF, xlsx, txt, md). Run **`index_rag.bat`** after adding or changing files (or enable RAG in setup to index automatically).
4. With **RAG_ENABLED=true**, only the most relevant chunks are sent to the model instead of every file.

## Quick CLI test (no hotkey)

```powershell
cd C:\Users\__USERNAME__\Documents\Data
python -c "
from dotenv import load_dotenv
load_dotenv()
from router.meta_router import complete
ans, meta = complete('Answer briefly.', 'What is Ohm law?')
print('tier:', meta.tier, 'provider:', meta.provider_id)
print(ans[:200])
"
```

## Tests

### 1. Short question → fast tier

- Select a short exam question (~50 chars), press Ctrl+B.
- Check `app.log` for `tier=fast` and a fast provider (e.g. `groq-llama`, `gemini-flash`).

### 2. Invalid Gemini key → failover

- Set `GEMINI_API_KEY=invalid` in `.env`, keep other keys valid.
- Ctrl+B on a short question.
- Expect success via Groq/HF/OpenRouter; log should not show gemini-flash success.

### 3. Rate limit cooldown (429)

- After Gemini quota errors, check `router/state.json` for `cooldown_until` on `gemini-flash`.
- Next request should skip Gemini and use another provider in logs.

### 4. Long context → balanced/reasoning

- Add a large PDF or text file to the Data folder.
- Ask a question with reasoning wording (e.g. "analyze the tradeoffs").
- Log should show `tier=balanced` or `tier=reasoning`.

### 5. Empty answer escalation

- If a provider returns empty text, log shows `outcome=empty` and router tries the next provider.
- Weak-answer escalation logs: `escalating fast -> balanced`.

## Logs

- App: `app.log`
- Provider cooldowns: `router/state.json`
