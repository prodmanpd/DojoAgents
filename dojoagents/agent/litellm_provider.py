"""LiteLLM SDK provider.

One provider for every backend LiteLLM can route to (Anthropic, Bedrock,
Vertex AI, Azure, Gemini, Mistral, Ollama, ...) plus LiteLLM Proxy (AI
gateway) deployments. LiteLLM returns OpenAI-shaped responses and stream
chunks, so streaming, reasoning deltas, tool-call assembly and usage
accounting are inherited from ``OpenAICompatibleProvider``; only the call
itself, model routing and credential handling differ.
"""

from __future__ import annotations

from typing import Any

from dojoagents.agent.context_length import ContextLengthExceededError, parse_context_length_error
from dojoagents.agent.models import LLMResult
from dojoagents.agent.providers import OpenAICompatibleProvider

PROXY_ROUTE = "litellm_proxy"


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - exercised via the error message
        raise ImportError("The litellm provider requires the optional dependency: " 'pip install "dojoagents[litellm]"') from exc
    return litellm


class LiteLLMProvider(OpenAICompatibleProvider):
    name = "litellm"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        author: str | None = None,
        extra_headers: dict[str, str] | None = None,
        completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            author=author,
            extra_headers=extra_headers,
        )
        # Drop per-provider unsupported params (e.g. parallel_tool_calls on
        # Anthropic, stream_options on some routes) instead of failing the turn.
        self.completion_kwargs: dict[str, Any] = {"drop_params": True, **(completion_kwargs or {})}

    @classmethod
    def from_config(cls, provider_cfg: Any) -> "LiteLLMProvider":
        return cls(
            api_key=provider_cfg.api_key,
            base_url=provider_cfg.base_url,
            author=provider_cfg.author,
            extra_headers=provider_cfg.extra_headers,
        )

    def _missing_credentials_result(self) -> LLMResult | None:
        # LiteLLM resolves provider credentials itself (ANTHROPIC_API_KEY,
        # AWS_*, GOOGLE_APPLICATION_CREDENTIALS, ...), and a proxy started
        # without a master key accepts unauthenticated requests.
        return None

    def _resolve_model(self, model: str) -> str:
        # The config loader splits "anthropic/claude-sonnet-4-5" into
        # author="anthropic" and model="claude-sonnet-4-5"; LiteLLM needs the
        # route back.
        resolved = model
        if self.author and not resolved.startswith(f"{self.author}/"):
            resolved = f"{self.author}/{resolved}"
        if self.base_url:
            route = resolved.split("/", 1)[0] if "/" in resolved else ""
            litellm = _import_litellm()
            # Behind a LiteLLM Proxy, a bare alias ("claude-sonnet") must be
            # sent through the proxy route or LiteLLM tries to infer a vendor
            # from the name. Explicit routes (azure/..., openai/...) keep the
            # base URL as their api_base.
            if route not in getattr(litellm, "provider_list", ()):
                resolved = f"{PROXY_ROUTE}/{resolved}"
        return resolved

    async def _create_completion(self, create_kwargs: dict[str, Any]) -> Any:
        litellm = _import_litellm()
        request = {**self.completion_kwargs, **create_kwargs}
        if request.get("tools") is None:
            request.pop("tools", None)
        if self.api_key:
            request["api_key"] = self.api_key
        if self.base_url:
            request["api_base"] = self.base_url
        if self.extra_headers:
            request["extra_headers"] = dict(self.extra_headers)
        try:
            return await litellm.acompletion(**request)
        except Exception as exc:
            context_error = getattr(getattr(litellm, "exceptions", None), "ContextWindowExceededError", None)
            if context_error is not None and isinstance(exc, context_error):
                max_context, requested = parse_context_length_error(str(exc))
                raise ContextLengthExceededError(str(exc), max_context=max_context, requested_tokens=requested) from exc
            raise


def litellm_context_info(model: str, base_url: str | None = None) -> tuple[int, tuple[str, ...]] | None:
    """Context window and input modalities from LiteLLM's model map.

    Returns None for names LiteLLM does not know (proxy aliases, custom
    deployments), so callers keep their existing fallbacks.
    """
    if base_url:
        return None
    try:
        litellm = _import_litellm()
        info = litellm.get_model_info(model)
    except Exception:
        return None
    context_window = info.get("max_input_tokens") or info.get("max_tokens")
    if not isinstance(context_window, int) or context_window <= 0:
        return None
    modalities = ("text", "image") if info.get("supports_vision") else ("text",)
    return context_window, modalities
