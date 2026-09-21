# Model Configuration

DojoAgents reads LLM provider configuration from:

```text
~/.dojo/agents.yaml
```

## Interactive Setup

```bash
dojoagents model
```

Custom config file:

```bash
dojoagents model --config ./agents.yaml
```

The command lets you choose a provider, set a base URL, enter an API key, probe available models, and save the result.

Probed models are saved as the provider's `models` candidates, while the
selected model remains its `model` default. The Dashboard exposes the current
configured choices through `GET /api/v1/models`.

## Rules

- Runtime code should read typed config through `ConfigStore.snapshot()`.
- Dashboard/API exposure must use redacted config.
- YAML may reference environment variables such as `${OPENAI_API_KEY}`.
- A provider can add request headers with an `extra_headers` string mapping;
  environment placeholders are supported in header values.

## LiteLLM

The `litellm` provider reaches every backend [LiteLLM](https://github.com/BerriAI/litellm)
supports (Anthropic, Bedrock, Vertex AI, Azure OpenAI, Gemini, Mistral, Ollama,
and more) through one provider, either in-process via the LiteLLM SDK or
through a LiteLLM Proxy. Install the optional extra first:

```bash
pip install "dojoagents[litellm]"
```

Direct, in-process: the model name carries the route, and each route reads
its provider's usual credentials (`ANTHROPIC_API_KEY`, `AWS_*`,
`GOOGLE_APPLICATION_CREDENTIALS`, ...), so `api_key` is optional.

```yaml
llm_provider:
  default: litellm
  providers:
    litellm:
      model: anthropic/claude-sonnet-4-5
      models:
        - anthropic/claude-sonnet-4-5
        - bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
        - vertex_ai/gemini-2.5-pro
```

Through a LiteLLM Proxy: set `base_url` to the proxy, use its virtual key, and
name the proxy's model aliases. Bare aliases are sent through the proxy route
automatically; `dojoagents model` (preset "LiteLLM Proxy") probes them.

```yaml
llm_provider:
  default: litellm
  providers:
    litellm:
      base_url: http://localhost:4000
      api_key_env: LITELLM_API_KEY
      model: claude-sonnet
```

Context windows and image support for known models come from LiteLLM's model
map, so compression thresholds and the Dashboard's image-input check follow the
selected model; `context_window` still overrides.
