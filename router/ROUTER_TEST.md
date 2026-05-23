# Meta-router smoke tests

## Setup

1. Run **`setup.bat`** in the project root.
   - Prompts for Python paths, optional ntfy notifications, and API keys.
   - Writes `.env`, installs requirements, then asks how to launch:
     - **1** Desktop shortcut (default), **2** Windows logon auto-start, or **3** manual only.
   - Only one method is configured; switching removes the other (e.g. logon task vs Desktop shortcut).
2. **Start:** double-click **Clip Assist** on your Desktop (or `scripts\start_clip_assist.vbs`).
3. **Stop logon task** if you used it before: `scripts\remove_from_startup.bat`.
4. Put study files in **`context/`** (PDF, xlsx, txt, md). Run **`scripts\index_rag.bat`** after adding or changing files.
5. With **RAG_ENABLED=true**, only the most relevant chunks are sent to the model instead of every file.
6. For development restarts: **`scripts\restart_service.bat`** (add `--nopause` to skip the final pause).

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

- Highlight a short question (~50 chars) and press Ctrl+B (selection is copied automatically).
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
