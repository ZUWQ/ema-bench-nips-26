"""
LLM configuration loader.

Reads OpenAI-compatible settings from environment variables or a JSON file.
Precedence: environment variables > config file > built-in defaults.

Copy ``llm_config.example.json`` to ``llm_config.json`` in this directory and fill in values.
``llm_config.json`` is listed in ``.gitignore`` — never commit real API keys or tenant-specific URLs.

Privacy: prefer ``OPENAI_*`` environment variables in CI and on shared machines so secrets are not
written to disk. If a key ever landed in Git history, rotate the key and rewrite history before
pushing to a public remote.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


class LLMConfig:
    """Loads and exposes LLM client settings."""

    # Built-in default when nothing else is set
    DEFAULT_MODEL = "gpt-5.4"

    # Environment variable names
    ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
    ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
    ENV_OPENAI_MODEL = "OPENAI_MODEL"
    ENV_OPENAI_CHAT_COMPLETION_EXTRA_KWARGS = "OPENAI_CHAT_COMPLETION_EXTRA_KWARGS"

    def __init__(self, config_file: Optional[Path] = None):
        """
        Args:
            config_file: Optional JSON path. Defaults to ``EmbodiedMAS/llm_config.json`` beside this module.
        """
        if config_file is None:
            config_file = Path(__file__).parent / "llm_config.json"

        self._config_file = config_file
        self._config_data: dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load JSON from disk when the file exists."""
        if self._config_file.exists():
            try:
                import json

                self._config_data = json.loads(self._config_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[LLMConfig] warning: could not read config file {self._config_file}: {e}")
                self._config_data = {}
        else:
            self._config_data = {}

    def get_api_key(self) -> Optional[str]:
        """
        API key resolution order:
        1. ``OPENAI_API_KEY`` environment variable
        2. ``api_key`` field in the JSON file
        3. ``None`` (client library default / error at call time)
        """
        api_key = os.getenv(self.ENV_OPENAI_API_KEY)
        if api_key:
            return api_key

        api_key = self._config_data.get("api_key")
        if api_key:
            return api_key

        return None

    def get_base_url(self) -> Optional[str]:
        """
        Base URL resolution order:
        1. ``OPENAI_BASE_URL``
        2. ``base_url`` in the JSON file
        3. ``None`` (vendor default endpoint)
        """
        base_url = os.getenv(self.ENV_OPENAI_BASE_URL)
        if base_url:
            return base_url

        base_url = self._config_data.get("base_url")
        if base_url:
            return base_url

        return None

    def get_model(self) -> str:
        """
        Model name resolution order:
        1. ``OPENAI_MODEL``
        2. ``model`` in the JSON file
        3. ``DEFAULT_MODEL``
        """
        model = os.getenv(self.ENV_OPENAI_MODEL)
        if model:
            return model

        model = self._config_data.get("model")
        if model:
            return model

        return self.DEFAULT_MODEL

    def get_client_kwargs(self) -> dict:
        """
        Keyword arguments for ``OpenAI(...)`` construction.

        Returns:
            Dict possibly containing ``api_key`` and/or ``base_url``.
        """
        kwargs = {}

        api_key = self.get_api_key()
        if api_key:
            kwargs["api_key"] = api_key

        base_url = self.get_base_url()
        if base_url:
            kwargs["base_url"] = base_url

        return kwargs

    def get_chat_completion_extra_kwargs(self) -> dict[str, Any]:
        """
        Extra kwargs merged into ``chat.completions.create``.

        Resolution order:
        1. ``OPENAI_CHAT_COMPLETION_EXTRA_KWARGS`` (JSON object)
        2. ``chat_completion_extra_kwargs`` in the JSON file
        3. Empty dict
        """
        raw_env = os.getenv(self.ENV_OPENAI_CHAT_COMPLETION_EXTRA_KWARGS)
        if raw_env:
            try:
                parsed = json.loads(raw_env)
                if isinstance(parsed, dict):
                    return parsed
                print(
                    f"[LLMConfig] warning: {self.ENV_OPENAI_CHAT_COMPLETION_EXTRA_KWARGS} is not a JSON object, ignored"
                )
            except Exception as e:
                print(
                    f"[LLMConfig] warning: could not parse {self.ENV_OPENAI_CHAT_COMPLETION_EXTRA_KWARGS}: {e}"
                )

        from_cfg = self._config_data.get("chat_completion_extra_kwargs")
        if isinstance(from_cfg, dict):
            return from_cfg
        return {}


_openai_patched: bool = False


def _merge_chat_completion_kwargs(
    original_kwargs: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge kwargs for ``chat.completions.create``:
    - Caller-supplied keys win (not overwritten)
    - ``extra_body`` dicts are shallow-merged with caller on top
    """
    merged = dict(original_kwargs)
    for key, val in extra_kwargs.items():
        if key == "extra_body":
            if "extra_body" not in merged:
                merged["extra_body"] = val
            elif isinstance(merged.get("extra_body"), dict) and isinstance(val, dict):
                merged["extra_body"] = {**val, **merged["extra_body"]}
            continue
        merged.setdefault(key, val)
    return merged


def _ensure_openai_chat_completion_patch() -> None:
    """Monkey-patch ``OpenAI`` so every client injects merged extra chat kwargs."""
    global _openai_patched
    if _openai_patched:
        return

    from openai import OpenAI

    original_init = OpenAI.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            raw_create = self.chat.completions.create
            if getattr(raw_create, "__embodiedmas_patched__", False):
                return

            def _wrapped_create(*c_args: Any, **c_kwargs: Any) -> Any:
                cfg = get_llm_config()
                extra = cfg.get_chat_completion_extra_kwargs()
                if extra:
                    c_kwargs = _merge_chat_completion_kwargs(c_kwargs, extra)
                return raw_create(*c_args, **c_kwargs)

            setattr(_wrapped_create, "__embodiedmas_patched__", True)
            self.chat.completions.create = _wrapped_create
        except Exception:
            # Non-fatal if the client layout does not support patching
            pass

    OpenAI.__init__ = _patched_init
    _openai_patched = True


_global_config: Optional[LLMConfig] = None


def get_llm_config(config_file: Optional[Path] = None) -> LLMConfig:
    """
    Singleton accessor for ``LLMConfig``.

    Args:
        config_file: Optional path; only honored on the first call.

    Returns:
        Shared ``LLMConfig`` instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = LLMConfig(config_file)
    _ensure_openai_chat_completion_patch()
    return _global_config


def create_openai_client(**kwargs: Any) -> Any:
    """
    Return an ``OpenAI``-compatible client that records token usage on ``chat.completions.create``.

    Use this when you do not want the global ``install()`` hook; do not combine both or counts may double.

    Requires the same import layout as this module (``EmbodiedMAS`` or repo root on ``sys.path``).
    """
    try:
        from Metric_Tool.llm_token_evaluation import (  # type: ignore
            ensure_atexit_flush_registered,
            record_chat_completion,
        )
    except ImportError:
        from EmbodiedMAS.Metric_Tool.llm_token_evaluation import (  # type: ignore
            ensure_atexit_flush_registered,
            record_chat_completion,
        )

    from openai import OpenAI

    ensure_atexit_flush_registered()
    inner = OpenAI(**kwargs)

    class _CompletionsProxy:
        def __init__(self, raw: Any) -> None:
            object.__setattr__(self, "_raw", raw)

        def create(self, *args: Any, **kw: Any) -> Any:
            completion = self._raw.create(*args, **kw)
            try:
                record_chat_completion(completion, kw.get("model"))
            except Exception:
                pass
            return completion

        def __getattr__(self, name: str) -> Any:
            return getattr(self._raw, name)

    class _ChatProxy:
        def __init__(self, raw: Any) -> None:
            object.__setattr__(self, "_raw", raw)

        @property
        def completions(self) -> Any:
            return _CompletionsProxy(self._raw.completions)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._raw, name)

    class _ClientProxy:
        def __init__(self, raw: Any) -> None:
            object.__setattr__(self, "_raw", raw)

        @property
        def chat(self) -> Any:
            return _ChatProxy(self._raw.chat)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._raw, name)

    return _ClientProxy(inner)


def reset_llm_config(config_file: Optional[Path] = None) -> LLMConfig:
    """
    Replace the global singleton (tests or hot reload).

    Args:
        config_file: Optional JSON path for the new instance.

    Returns:
        Fresh ``LLMConfig``.
    """
    global _global_config
    _global_config = LLMConfig(config_file)
    return _global_config
