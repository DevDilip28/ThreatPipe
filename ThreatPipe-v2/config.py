"""
ThreatPipe v2 — Config loader
==============================
config.yaml vaachto aani LLM + settings return karto.
Har setting load hoil tevha log madhe print hotil.
"""

import os
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # minimal fallback when langchain not installed
    class BaseCallbackHandler:  # type: ignore
        pass

try:
    from langchain_mistralai import ChatMistralAI
except Exception:
    ChatMistralAI = None  # type: ignore

from logger import get_logger, log_llm_response

log = get_logger(__name__)

# ─── Load YAML ────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def _load_yaml() -> dict:
    if not _CONFIG_PATH.exists():
        log.warning(f"config.yaml not found at {_CONFIG_PATH} — using defaults")
        return {}
    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    log.debug(f"Config loaded from {_CONFIG_PATH}")
    log.debug(f"  llm.model        : {cfg.get('llm', {}).get('model')}")
    log.debug(f"  graph.db_path    : {cfg.get('graph', {}).get('db_path')}")
    log.debug(f"  agent.max_cycles : {cfg.get('agent', {}).get('max_cycles')}")
    log.debug(f"  web_root         : {cfg.get('web_root')}")
    return cfg

CONFIG = _load_yaml()


# ─── Token cost callback ──────────────────────────────────────────────────────

class TokenCostManager(BaseCallbackHandler):
    # Mistral Small pricing (as of mid 2025)
    COST_PER_1K_INPUT  = 0.0001
    COST_PER_1K_OUTPUT = 0.0003

    def __init__(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost          = 0.0
        self._call_count         = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self._call_count += 1
        usage    = response.llm_output.get("token_usage", {})
        prompt_t = usage.get("prompt_tokens",     0)
        comp_t   = usage.get("completion_tokens", 0)
        total    = prompt_t + comp_t

        if total > 0:
            cost = (
                (prompt_t * self.COST_PER_1K_INPUT  / 1000) +
                (comp_t   * self.COST_PER_1K_OUTPUT / 1000)
            )
            self.total_input_tokens  += prompt_t
            self.total_output_tokens += comp_t
            self.total_cost          += cost

            # Get response text for log
            try:
                resp_text = response.generations[0][0].text if response.generations else ""
            except Exception:
                resp_text = ""

            model_name = CONFIG.get("llm", {}).get("model", "unknown")
            log_llm_response(
                log,
                model      = model_name,
                response_text = resp_text,
                input_tokens  = prompt_t,
                output_tokens = comp_t,
                cost_usd      = cost,
            )

    def summary(self) -> dict:
        return {
            "calls":         self._call_count,
            "input_tokens":  self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }


# Single shared instance so all calls accumulate
token_manager = TokenCostManager()


# ─── LLM factory ─────────────────────────────────────────────────────────────
# New: router-first. Legacy mistral path kept for backward compat.

def get_router():
    """Return shared LLMRouter (Jarvislabs -> NVIDIA -> Groq -> local -> ...)."""
    from llm.router import get_router as _gr
    return _gr()


def get_chat_model(task_type="NORMAL", temperature=0.0, max_tokens=2000):
    """Provider-independent chat model for a given task type."""
    from llm.base import TaskType as _TT
    from llm.langchain_adapter import get_chat_model as _gcm
    try:
        tt = _TT(task_type) if isinstance(task_type, str) else task_type
    except Exception:
        from llm.base import TaskType as _TT2
        tt = _TT2.NORMAL
    return _gcm(task_type=tt, temperature=temperature, max_tokens=max_tokens)


def get_llm():
    llm_cfg = CONFIG.get("llm", {})
    provider = llm_cfg.get("provider", "mistral")

    # Explicit router request, or mistral key missing -> use router
    use_router = (
        provider in ("router", "multi", "auto")
        or not os.getenv("MISTRAL_API_KEY")
        or os.getenv("LLM_FORCE_ROUTER", "") == "1"
    )
    if use_router:
        try:
            r = get_router()
            avail = r.available_providers()
            log.info(f"Loading LLM via router  mode={r.mode.value}  available={avail}")
            if not avail:
                log.error("No LLM providers configured (set JARVISLABS_API_KEY/NVIDIA_API_KEY/GROQ_API_KEY or start local server)")
                raise ValueError("No LLM providers configured.")
            # default NORMAL; callers can use get_chat_model(TASK) for better routing
            from llm.base import TaskType
            return get_chat_model(TaskType.NORMAL,
                                  temperature=llm_cfg.get("temperature", 0.0),
                                  max_tokens=llm_cfg.get("max_tokens", 2000))
        except Exception as exc:
            log.warning(f"Router init failed ({exc}); falling back to legacy mistral if possible")

    log.info(f"Loading LLM  provider={provider}  model={llm_cfg.get('model')}")

    if provider == "mistral":
        if ChatMistralAI is None:
            raise ValueError("langchain_mistralai not installed; set LLM provider=router instead.")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            log.error("MISTRAL_API_KEY environment variable not set!")
            raise ValueError("MISTRAL_API_KEY not set.")

        llm = ChatMistralAI(
            model       = llm_cfg.get("model", "mistral-small-latest"),
            api_key     = api_key,
            temperature = llm_cfg.get("temperature", 0.0),
            max_tokens  = llm_cfg.get("max_tokens", 1000),
            callbacks   = [token_manager],
        )
        log.info(f"  ✓ MistralAI ready  model={llm_cfg.get('model')}")
        return llm

    elif provider == "local_llama":
        # Legacy alias -> route to local/Ollama via router (no recursion)
        log.info("local_llama requested -> using router BULK/local fallback")
        return get_chat_model("BULK_PROCESSING",
                              temperature=llm_cfg.get("temperature", 0.0),
                              max_tokens=llm_cfg.get("max_tokens", 2000))

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ─── Helper accessors ─────────────────────────────────────────────────────────

def get_tool_path(tool_name: str) -> str:
    """Return configured path for a SIFT tool, falling back to tool name."""
    path = CONFIG.get("tools", {}).get(tool_name, tool_name)
    log.debug(f"Tool path lookup: {tool_name} → {path}")
    return path

def get_web_root() -> str:
    return CONFIG.get("web_root", "/tmp/threatpipe_evidence/www")

def get_db_path() -> str:
    return CONFIG.get("graph", {}).get("db_path", "./threatpipe_evidence.db")

def get_max_cycles() -> int:
    return CONFIG.get("agent", {}).get("max_cycles", 3)

def get_confidence_threshold() -> float:
    return CONFIG.get("agent", {}).get("confidence_threshold", 0.7)