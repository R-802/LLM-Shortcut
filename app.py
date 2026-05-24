"""
Headless hotkey service: Ctrl+B copies the current selection, queries LLM via meta-router,
and copies the answer to the clipboard.

Uses Windows RegisterHotKey (no low-level keyboard hook).
Run at logon (hidden): pythonw.exe app.py
"""

from dotenv import load_dotenv
import requests
import pyperclip
from pathlib import Path
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging
import threading
import time
import sys
import os
import ctypes
from router.sanitize import sanitize_answer
from router.rag import (
    is_enabled as rag_enabled,
    retrieve as rag_retrieve,
    start_index_watcher as rag_start_index_watcher,
    warmup as rag_warmup,
)
from router.meta_router import complete
from router.documents import extract_text as extract_document_text, read_pdf
from router.images import (
    ImageAttachment,
    attachment_from_clipboard,
    collect_matching_from_dir,
)
from router.rag_sources import infer_source_filters, list_context_image_names
from router.documents import iter_source_files
from router.hotkey import load_hotkey_from_env
from router.stdio import patch_pythonw_stdio

patch_pythonw_stdio()


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

load_dotenv(APP_DIR / ".env")

LOG_PATH = APP_DIR / "app.log"
CONTEXT_PATH = APP_DIR / "context"
REQUEST_TIMEOUT_S = max(30, int(os.environ.get("REQUEST_TIMEOUT_S", "90")))


def _context_fallback_max_chars() -> int:
    try:
        return max(2000, int(os.environ.get("CONTEXT_FALLBACK_MAX_CHARS", "12000")))
    except ValueError:
        return 12000


def _log_prompt_max_chars() -> int:
    try:
        return max(500, int(os.environ.get("LOG_PROMPT_MAX_CHARS", "8000")))
    except ValueError:
        return 8000


def _log_llm_request(
    question: str,
    user_prompt: str,
    context_files: list[str],
    context_mode: str,
    image_labels: list[str],
) -> None:
    """Log the question, attached files, and full user prompt sent to the model."""
    max_log = _log_prompt_max_chars()
    if len(user_prompt) <= max_log:
        prompt_for_log = user_prompt
    else:
        prompt_for_log = (
            user_prompt[:max_log]
            + f"\n\n[... prompt truncated for log ({len(user_prompt)} chars total) ...]"
        )
    files_str = ", ".join(context_files) if context_files else "(none)"
    images_str = ", ".join(image_labels) if image_labels else "(none)"
    logging.info("Request question: %s", question)
    logging.info("Context files: %s", files_str)
    logging.info("Context images: %s", images_str)
    logging.info("Context mode: %s", context_mode)
    logging.info("User prompt (%d chars):\n%s",
                 len(user_prompt), prompt_for_log)


NTFY_MAX_BODY = 3500
APP_NAME = "Clip Assist"
APP_BUILD = "2026-05-24-topic-filter-v2"
HOTKEY_ID = 1
MUTEX_NAME = "Global\\ClipAssistSvc"
ERROR_ALREADY_EXISTS = 183

WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000

user32 = ctypes.windll.user32

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return default


NTFY_ENABLED = _env_bool("NTFY_ENABLED", default=True)
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
if NTFY_ENABLED and not NTFY_TOPIC:
    logging.error(
        "NTFY_ENABLED is true but NTFY_TOPIC is empty. Set NTFY_TOPIC or set NTFY_ENABLED=false in %s",
        APP_DIR / ".env",
    )
    sys.exit(1)
if not NTFY_ENABLED:
    logging.info("ntfy notifications disabled (NTFY_ENABLED=false).")


_base_instruction = (
    "Respond in the style of a student writing a high-scoring exam answer.\n"
    "Use clear, direct language and standard academic terminology appropriate to the subject.\n"
    "Structure answers logically (e.g., brief paragraphs or bullet points where appropriate) without unnecessary elaboration.\n"
    "Avoid repetition, filler content, and background explanation unless it directly contributes to marks.\n"
    "Be very concise but complete: ensure all essential steps, definitions, or reasoning required for are included.\n"
    "Do not include commentary, teaching explanations, or meta-explanations.\n"
    "Do not include elaborations, examples, or non-essential details.\n"
    "Output plain text only with no formatting at all beyond basic line breaks and simple lists.\n"
    "Do not use markdown formatting.\n"
    "Do not use asterisks for headings or emphasis (use * only for multiplication in formulas).\n"
    "Do not use bold or italic formatting.\n"
    "Do not use quotes.\n"
    "Do not use code blocks.\n"
    "Do not use tables.\n"
    "Do not include images in your reply (text only).\n"
    "If images are attached, use them to answer the question.\n"
    "Do not use links.\n"
    "Do not use footnotes.\n"
    "Do not use endnotes.\n"
    "For maths use only ASCII operators: +, -, *, /, and powers as x^n (e.g. r^2, v^3).\n"
    "Do not use Unicode maths symbols (no pi, rho, multiply sign, divide sign, square root sign, etc.).\n"
    "Spell Greek letters in words (pi, rho, eta) - never use Greek symbol characters.\n"
    "Do not use LaTeX, markdown, or special formatting.\n"
    "Use compact units only: MW, kW, kWh, kJ/kg, t/h, deg C, %. "
    "Do not spell out megawatt, kilowatt, kilojoule, or degrees Celsius.\n"
    "Show each calculation once in a short line. Do not repeat wrong attempts, "
    "unit conversion loops, or self-corrections.\n"
    "Use New Zealand styled English.\n"
    "CONTEXT RULES:\n"
    "- If context is provided, use it as the primary source of data, constants, and method.\n"
    "- If context includes a worked example or solution sheet, apply the same formula and approach "
    "to the current question with the given data values. Do not switch to a different method.\n"
    "- Use exact constants from context (e.g. heat capacity, efficiencies, conversion factors). "
    "Do not substitute generic textbook values if the context provides them.\n"
    "- If specific numeric data (e.g. solar resource, flow rate) is not in context, state that "
    "clearly and work with the data given in the question only.\n"
    "- If context is provided but covers a different scenario (e.g. a hotel vs a school), "
    "still use its formula structure and constants for this question.\n"
    "- Do not say 'the exact value is not provided' if context gives a method to calculate it.\n"
    "- If context is labelled WORKED EXAMPLE, use only its formula/method; "
    "do NOT use its numerical results (area, volume, demand) as the answer.\n"
    "COURSE CONSTANTS (RESE321 — use these unless the question states otherwise):\n"
    "- Water volumetric heat capacity: 1.160 kWh/m3/K (= 4176 kJ/m3/K).\n"
    "  Pool/tank energy: Q = volume_m3 * 1.160 * delta_T_K (kWh). "
    "This gives daily energy needed to maintain temperature against ambient losses.\n"
    "- Flat plate collector efficiency: ~0.63 (use coefficient method from lecture 3-2 if given).\n"
    "- Evacuated tube collector efficiency: ~0.66.\n"
    "- Solar resource for area calc: use the MINIMUM cumulative daily resource (kWh/m2/day), "
    "not the monthly average, to ensure the system works on the worst day.\n"
    "- Collector area formula: A = Q_daily / (resource_min * eta) in m2.\n"
    "- Do NOT use 4.184 kJ/kg/K for pool/tank volume problems — use 1.160 kWh/m3/K instead.\n"
)

_busy = threading.Lock()
_llm_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="clip-assist-llm"
)


def _ensure_single_instance() -> None:
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW(None, True, MUTEX_NAME)
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        logging.error(
            "Another instance is already running. End pythonw.exe in Task Manager, then restart."
        )
        sys.exit(0)


def _ntfy_send(title: str, message: str, priority: str) -> None:
    if not NTFY_ENABLED:
        return
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    body = message if len(
        message) <= NTFY_MAX_BODY else message[:NTFY_MAX_BODY] + "\n..."
    headers = {"Title": title[:250], "Priority": priority, "Tags": "bell"}
    try:
        resp = requests.post(
            url,
            data=body.encode("utf-8"),
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        logging.info("ntfy sent: %s (HTTP %s)", title, resp.status_code)
    except requests.RequestException as exc:
        logging.error("ntfy failed (%s): %s", title, exc)


def ntfy_notify(title: str, message: str, priority: str = "3") -> None:
    """Push notification via ntfy.sh (runs in background thread)."""
    if not NTFY_ENABLED:
        return
    threading.Thread(
        target=_ntfy_send,
        args=(title, message, priority),
        daemon=True,
    ).start()


def _load_files_from_dir(folder: Path, file_data: str) -> str:
    skip_names = {
        "app.py", "app.log", ".env", "requirements.txt",
        "ROUTER_TEST.md", ".env.example",
    }
    skip_suffixes = {".pyc", ".yaml", ".md", ".bat", ".ps1", ".txt"}
    if not folder.is_dir():
        return file_data

    for file_path in folder.iterdir():
        if not file_path.is_file() or file_path.name in skip_names:
            continue
        if file_path.suffix in skip_suffixes or file_path.name.startswith("."):
            continue

        filename = file_path.name
        if filename.endswith(".pdf"):
            content = read_pdf(file_path)
            file_data += f"\n\nFile ({filename}):\n{content}"

        elif filename.endswith(".xlsx"):
            try:
                text = extract_document_text(file_path)
                file_data += f"\n\nFile ({filename}):\n{text}"
            except Exception:
                file_data += f"\n\nFile ({filename}) could not be read."

        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_data += f"\n\nFile ({filename}):\n" + f.read()
            except Exception:
                file_data += f"\n\nFile ({filename}) could not be read."

    return file_data


def _truncate_context(text: str, max_chars: int, label: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    logging.warning(
        "%s truncated from %d to %d chars (raise CONTEXT_FALLBACK_MAX_CHARS in .env).",
        label,
        len(text),
        max_chars,
    )
    return text[:max_chars] + "\n\n[... context truncated ...]"


def load_context_files() -> str:
    """Load text from context/ files (fallback when RAG returns nothing)."""
    if not CONTEXT_PATH.is_dir():
        return ""
    raw = _load_files_from_dir(CONTEXT_PATH, "")
    return _truncate_context(raw, _context_fallback_max_chars(), "Fallback context")


def load_prompt_context(question: str) -> tuple[str, str, list[str]]:
    """
    Load context for the prompt.

    Returns (context_text, mode, source_paths):
      - rag: top-K chunks from ChromaDB for this question
      - fallback: capped dump of context/ files (RAG off or failed)
      - off: RAG disabled and no fallback
      - empty: no files / no matches
    """
    if rag_enabled() and CONTEXT_PATH.is_dir():
        retrieved, status, sources = rag_retrieve(
            question, CONTEXT_PATH, APP_DIR)
        if retrieved.strip():
            return retrieved, "rag", sources
        if status == "below_threshold":
            logging.info(
                "RAG: question unrelated to context/; no file context attached."
            )
            return "", "empty", []
        if status == "named_file_missing":
            logging.warning(
                "RAG: named document(s) could not be loaded; no file context attached."
            )
            return "", "empty", sources
        if status == "busy":
            logging.warning(
                "RAG: index busy (rebuild in progress); no file context attached."
            )
            return "", "empty", []
        if status == "empty_index":
            logging.info(
                "RAG index missing; falling back to capped context/ files.")
        else:
            logging.info(
                "RAG found no chunks; falling back to capped context/ files.")
        fallback = load_context_files()
        if fallback.strip():
            all_files = [
                p.relative_to(CONTEXT_PATH).as_posix()
                for p in iter_source_files(CONTEXT_PATH)
            ]
            return fallback, "fallback", all_files
        return "", "empty", []
    if not rag_enabled() and CONTEXT_PATH.is_dir():
        fallback = load_context_files()
        if fallback.strip():
            all_files = [
                p.relative_to(CONTEXT_PATH).as_posix()
                for p in iter_source_files(CONTEXT_PATH)
            ]
            return fallback, "fallback", all_files
        return "", "empty", []
    return "", "off", []


def load_context_images(question: str, text_filters: list[str] | None = None) -> list[ImageAttachment]:
    """Attach context/ images only when the filename matches the question."""
    if not CONTEXT_PATH.is_dir():
        return []
    if text_filters is None:
        text_sources = [
            p.relative_to(CONTEXT_PATH).as_posix()
            for p in iter_source_files(CONTEXT_PATH)
        ]
        image_sources = list_context_image_names(CONTEXT_PATH)
        text_filters = infer_source_filters(
            question, text_sources + image_sources)
    images = collect_matching_from_dir(CONTEXT_PATH, question, text_filters)
    if images:
        names = ", ".join(img.label for img in images)
        logging.info("Context images attached (filename match): %s", names)
    elif list_context_image_names(CONTEXT_PATH):
        logging.info(
            "Context images skipped: no filename match for this question."
        )
    return images


def send_ctrl_c() -> None:
    """Copy the current selection to the clipboard (Windows Ctrl+C)."""
    VK_CONTROL = 0x11
    VK_C = 0x43
    KEYEVENTF_KEYUP = 0x0002
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(VK_C, 0, 0, 0)
    time.sleep(0.02)
    user32.keybd_event(VK_C, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.1)


def capture_from_clipboard() -> tuple[str, list[ImageAttachment]]:
    """
    Copy the current selection (Ctrl+C), then read text and/or image from the clipboard.

    When text is captured, clipboard images are ignored so an old screenshot does not
    override the selection. For image-only questions, use Win+Shift+S first, or press
    the hotkey with only an image already on the clipboard and nothing selected.
    """
    send_ctrl_c()
    text = pyperclip.paste().strip()

    images: list[ImageAttachment] = []
    if not text:
        clip_image = attachment_from_clipboard(label="clipboard_image")
        if clip_image:
            images.append(clip_image)
            logging.info("Captured clipboard image (%s).", clip_image.label)

    if text:
        logging.info("Captured %d chars of clipboard text.", len(text))
    return text, images


def ask_llm(question: str, images: list[ImageAttachment] | None = None) -> tuple[str, str]:
    """Return (answer, provider_id) via meta-router."""
    text_sources = []
    image_sources: list[str] = []
    if CONTEXT_PATH.is_dir():
        text_sources = [
            p.relative_to(CONTEXT_PATH).as_posix()
            for p in iter_source_files(CONTEXT_PATH)
        ]
        image_sources = list_context_image_names(CONTEXT_PATH)
    source_filters = infer_source_filters(
        question, text_sources + image_sources
    )

    file_data, context_mode, context_files = load_prompt_context(question)
    user = f"Question:\n{question}\n\nContext:\n{file_data}"

    all_images: list[ImageAttachment] = []
    all_images.extend(load_context_images(question, source_filters))
    if images:
        for img in images:
            if not any(i.label == img.label for i in all_images):
                all_images.append(img)
    if len(all_images) > 6:
        all_images = all_images[:6]

    image_labels = [img.label for img in all_images]
    _log_llm_request(question, user, context_files, context_mode, image_labels)

    answer, meta = complete(
        system=_base_instruction,
        user_prompt=user,
        images=all_images or None,
    )
    answer = sanitize_answer(answer.strip())
    logging.info(
        "Routed tier=%s provider=%s model=%s attempts=%d escalated=%s "
        "instruction_chars=%d context_chars=%d context_mode=%s images=%d",
        meta.tier,
        meta.provider_id,
        meta.litellm_model,
        meta.attempts,
        meta.escalated,
        len(_base_instruction),
        len(file_data),
        context_mode,
        len(all_images),
    )
    return answer, meta.provider_id


def _handle_hotkey() -> None:
    if not _busy.acquire(blocking=False):
        logging.info("Hotkey ignored; previous request still running.")
        ntfy_notify(
            APP_NAME,
            "Still working on the previous question. Wait for the answer or a timeout message.",
            priority="3",
        )
        return

    try:
        question, clip_images = capture_from_clipboard()

        if not question and not clip_images:
            pyperclip.copy("No text or image selected.")
            logging.warning("Empty selection.")
            ntfy_notify(APP_NAME,
                        "No text or image selected.", priority="3")
            return

        if not question:
            question = "Answer the question shown in the image(s)."

        context_images = load_context_images(question)
        has_images = bool(clip_images or context_images)

        preview = question[:120] + ("…" if len(question) > 120 else "")
        # ntfy_notify(APP_NAME, f"Working on:\n{preview}", priority="1")

        logging.info(
            "Sending %d chars, %d image(s) to meta-router.",
            len(question),
            len(clip_images) + len(context_images),
        )
        future = _llm_executor.submit(ask_llm, question, clip_images)
        try:
            answer, provider_id = future.result(timeout=REQUEST_TIMEOUT_S)
        except FuturesTimeoutError:
            logging.error("LLM request timed out after %ds", REQUEST_TIMEOUT_S)
            pyperclip.copy(
                f"Request timed out after {REQUEST_TIMEOUT_S}s. Try again."
            )
            ntfy_notify(
                APP_NAME,
                f"Timed out after {REQUEST_TIMEOUT_S}s. Wait a moment, then try again.",
                priority="4",
            )
            return
        pyperclip.copy(answer or "No response from model.")
        logging.info("Answer copied (%d chars) via %s.",
                     len(answer), provider_id)

        footer = f"\n\n— via {provider_id}"
        body = (answer or "No response from model. Paste from clipboard on your PC.")
        if len(body) + len(footer) <= NTFY_MAX_BODY:
            body += footer
        ntfy_notify("Answer ready", body, priority="max")
    except Exception as exc:
        pyperclip.copy(f"Error: {exc}")
        logging.exception("Hotkey handler failed.")
        ntfy_notify(f"{APP_NAME} failed", str(exc), priority="5")
    finally:
        _busy.release()


def _on_hotkey() -> None:
    threading.Thread(target=_handle_hotkey, daemon=True).start()


def main() -> None:
    _ensure_single_instance()
    logging.info("Build %s | python=%s | cwd=%s",
                 APP_BUILD, sys.executable, os.getcwd())

    try:
        hotkey_mod, hotkey_vk, hotkey_display = load_hotkey_from_env()
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    modifiers = hotkey_mod | MOD_NOREPEAT
    if not user32.RegisterHotKey(None, HOTKEY_ID, modifiers, hotkey_vk):
        logging.error(
            "RegisterHotKey(%s) failed. Another program may already own this shortcut.",
            hotkey_display,
        )
        sys.exit(1)

    logging.info(
        "Hotkey service started (%s via RegisterHotKey).", hotkey_display)
    ntfy_notify(
        f"{APP_NAME} started",
        f"Build {APP_BUILD} is running. {hotkey_display} is active.",
        "4",
    )

    if rag_enabled():
        rag_start_index_watcher(CONTEXT_PATH, APP_DIR)
        threading.Thread(
            target=rag_warmup,
            args=(APP_DIR, CONTEXT_PATH),
            daemon=True,
            name="rag-warmup",
        ).start()

    msg = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                _on_hotkey()
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    finally:
        user32.UnregisterHotKey(None, HOTKEY_ID)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
