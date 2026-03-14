"""
OpenAI LLM wrapper with customizable base_url.
Compatible with legacy langchain OpenAI interfaces.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from urllib.parse import urlsplit

from openai import OpenAI, AzureOpenAI

# Import centralized constants configuration
try:
    # Method 1: relative import (normal package usage)
    from ..config import constants as constants
except (ImportError, ValueError):
    try:
        # Method 2: absolute import (fallback)
        from mansion.config import constants as constants
    except ImportError:
        # Method 3: load constants file directly (standalone module execution)
        import os
        constants_path = os.path.join(os.path.dirname(__file__), "..", "config", "constants.py")
        if os.path.exists(constants_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("constants", constants_path)
            constants = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(constants)
        else:
            # If all methods fail, provide basic default config
            class _Constants:
                LLM_PROVIDER = "openai"
                OPENAI_API_KEY = None
                AZURE_OPENAI_API_KEY = None
                LLM_MODEL_NAME = "gpt-4o-2024-05-13"
                OPENAI_API_BASE = None
                AZURE_OPENAI_ENDPOINT = None
                AZURE_OPENAI_DEPLOYMENT = None
                AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
                LLM_MAX_TOKENS = 2048
                LLM_TEMPERATURE = 0.7
                LLM_TIMEOUT = 120.0
                LLM_MAX_RETRIES = 3
            constants = _Constants()


def _sanitize_url_for_logs(url: Any) -> str:
    """Redact non-essential endpoint details before printing to logs."""
    if not url:
        return "N/A"

    try:
        parsed = urlsplit(str(url))
    except Exception:
        return "<configured>"

    if not parsed.scheme or not parsed.netloc:
        return "<configured>"

    host = parsed.hostname or parsed.netloc
    if host == "api.openai.com":
        safe_host = host
    else:
        parts = host.split(".")
        if len(parts) >= 2:
            safe_host = f"***.{'.'.join(parts[-2:])}"
        else:
            safe_host = "***"

    port = f":{parsed.port}" if parsed.port else ""
    path_hint = "/..." if parsed.path and parsed.path != "/" else ""
    return f"{parsed.scheme}://{safe_host}{port}{path_hint}"


class OpenAIWrapper:
    """
    OpenAI API wrapper compatible with langchain.llms.OpenAI.
    Supports:
    - OpenAI official API
    - Azure OpenAI
    - Third-party OpenAI-compatible APIs
    - Multiprocessing (pickle-safe)

    Config priority: constructor args > constants.py
    Key source: constants.py (loaded from environment variables)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        # Azure OpenAI-specific parameters
        use_azure: Optional[bool] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        profile: Optional[str] = None,  # "openai" or "azure"
        node_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI or Azure OpenAI client.

        Config priority: args > profile/node_name > constants.py

        Args:
            model_name: Model name
            max_tokens: Max token count
            temperature: Temperature
            openai_api_key: API key
            openai_api_base: API base URL
            headers: Extra request headers
            timeout: Request timeout
            max_retries: Max retries
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name
            api_version: Azure API version
            profile: Explicit config profile ("openai" or "azure")
            node_name: Node name for profile auto-selection from node_config.json
        """
        # 1. Resolve profile
        selected_profile = profile
        if node_name:
            node_cfg_path = Path(__file__).parent.parent / "config" / "node_config.json"
            if node_cfg_path.exists():
                try:
                    with open(node_cfg_path, "r", encoding="utf-8") as f:
                        node_map = json.load(f)
                    selected_profile = node_map.get(node_name, profile)
                except Exception as e:
                    print(f"[OpenAIWrapper] Failed to load node_config.json: {e}")

        # Get base profile config
        base_profile_config = {}
        if selected_profile == "openai":
            base_profile_config = constants.OPENAI_CONFIG
        elif selected_profile == "azure":
            base_profile_config = constants.AZURE_CONFIG
        elif selected_profile == "azure_gpt5":
            base_profile_config = getattr(constants, "AZURE_GPT5_CONFIG", constants.AZURE_CONFIG)

        # 2. Merge config by priority
        # Priority: args > profile > constants.py
        
        self.model_name = model_name or base_profile_config.get("model") or constants.LLM_MODEL_NAME
        self.max_tokens = max_tokens or base_profile_config.get("max_tokens") or constants.LLM_MAX_TOKENS
        self.temperature = temperature if temperature is not None else base_profile_config.get("temperature", constants.LLM_TEMPERATURE)

        # Resolve provider
        if use_azure is not None:
            self.use_azure = use_azure
        elif selected_profile and selected_profile.startswith("azure"):
            self.use_azure = True
        elif selected_profile == "openai":
            self.use_azure = False
        else:
            self.use_azure = constants.LLM_PROVIDER == "azure"

        # Resolve keys and endpoints
        if self.use_azure:
            api_key = openai_api_key or base_profile_config.get("api_key") or constants.AZURE_OPENAI_API_KEY
            azure_ep = azure_endpoint or base_profile_config.get("endpoint") or constants.AZURE_OPENAI_ENDPOINT
            azure_dep = azure_deployment or base_profile_config.get("deployment") or constants.AZURE_OPENAI_DEPLOYMENT
            api_ver = api_version or base_profile_config.get("api_version") or constants.AZURE_OPENAI_API_VERSION
            
            if not all([api_key, azure_ep, azure_dep]):
                raise ValueError(f"Incomplete Azure OpenAI config (Node: {node_name}, Profile: {selected_profile})")

            # Keep behavior aligned with test_azure_gpt51.py: no extra endpoint rewriting.
            # Proxy services (e.g., V-API) may depend on path segments like /v1.
            self._init_kwargs = {
                "api_key": api_key,
                "azure_endpoint": azure_ep,
                "api_version": api_ver,
                "timeout": timeout or constants.LLM_TIMEOUT,
                "max_retries": max_retries or constants.LLM_MAX_RETRIES,
            }
            self.azure_deployment = azure_dep
        else:
            api_key = openai_api_key or base_profile_config.get("api_key") or constants.OPENAI_API_KEY
            api_base = openai_api_base or base_profile_config.get("base_url") or constants.OPENAI_API_BASE
            
            if not api_key:
                raise ValueError(f"OpenAI API key missing (Node: {node_name}, Profile: {selected_profile})")

            self._init_kwargs = {
                "api_key": api_key,
                "timeout": timeout or constants.LLM_TIMEOUT,
                "max_retries": max_retries or constants.LLM_MAX_RETRIES,
            }
            if api_base:
                self._init_kwargs["base_url"] = api_base
        
        # Add custom request headers
        if headers:
            self._init_kwargs["default_headers"] = headers

        # Create client (lazy init)
        self._client = None
        
        # Print final config
        print(f"\n{'='*70}")
        print(f"[OpenAIWrapper] Initialized | Node: {node_name or 'N/A'} | Profile: {selected_profile or 'auto'}")
        print(f"{'='*70}")
        print(f"Provider: {'Azure OpenAI' if self.use_azure else 'OpenAI-compatible API'}")
        if self.use_azure:
            print(f"Azure Deployment: {self.azure_deployment}")
            print(f"Azure Endpoint: {_sanitize_url_for_logs(self._init_kwargs.get('azure_endpoint'))}")
        else:
            print(f"Model: {self.model_name}")
            base_url = self._init_kwargs.get('base_url', 'https://api.openai.com/v1')
            print(f"Base URL: {_sanitize_url_for_logs(base_url)}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        print(f"{'='*70}\n")
    
    def _create_client(self):
        """Create or rebuild OpenAI/Azure OpenAI client."""
        if self.use_azure:
            self._client = AzureOpenAI(**self._init_kwargs)
        else:
            self._client = OpenAI(**self._init_kwargs)
    
    @property
    def client(self):
        """Lazy-load client with multiprocessing compatibility."""
        if self._client is None:
            self._create_client()
        return self._client
    
    def __getstate__(self):
        """Pickle support: exclude non-serializable client when saving state."""
        state = self.__dict__.copy()
        state['_client'] = None
        return state
    
    def __setstate__(self, state):
        """Pickle support: restore state and rebuild client lazily."""
        self.__dict__.update(state)

    def __call__(self, prompt: str, stop: Optional[list] = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, stop=stop, **kwargs)

    def chat(self, messages: List[Dict[str, Any]], stop: Optional[list] = None, **kwargs) -> str:
        """
        Call LLM to generate text. Supports full messages payload (including multimodal).
        """
        import time
        import openai

        model_to_use = self.azure_deployment if self.use_azure else self.model_name
        
        # Initialize base params first so request_kwargs is always defined
        request_kwargs = {
            "model": model_to_use,
            "messages": messages,
        }

        # Heuristic check for reasoning models (use max_completion_tokens; often no temperature/stop)
        model_lower = model_to_use.lower()
        is_reasoning_model = any(m in model_lower for m in ["gpt-5", "o1-", "o3-"])
        
        if is_reasoning_model:
            request_kwargs["max_completion_tokens"] = self.max_tokens
        else:
            request_kwargs["max_tokens"] = self.max_tokens
            request_kwargs["temperature"] = self.temperature
            if stop:
                request_kwargs["stop"] = stop
        
        request_kwargs.update(kwargs)

        def _preview_length(msgs: List[Dict[str, Any]]) -> int:
            total = 0
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            total += len(part.get("text", ""))
                else:
                    total += len(str(content))
            return total

        print(f"\n{'='*60}")
        print(f"[LLM] Call start | Model: {request_kwargs['model']} | Message count: {len(messages)} | Estimated input length: {_preview_length(messages)}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                # Auto-fix retry for "max_tokens is not supported" errors
                error_msg = str(e)
                if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                    print(f"[LLM] Auto-switching max_tokens -> max_completion_tokens and retrying...")
                    request_kwargs.pop("max_tokens", None)
                    request_kwargs["max_completion_tokens"] = self.max_tokens
                    # Reasoning models may also reject temperature and other fields
                    if "temperature" in error_msg:
                        request_kwargs.pop("temperature", None)
                    response = self.client.chat.completions.create(**request_kwargs)
                else:
                    raise e

            elapsed = time.time() - start_time
            result = response.choices[0].message.content

            print(f"[LLM] Call succeeded | Elapsed: {elapsed:.2f}s | Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
            print(f"Response preview: {result[:200]}..." if result and len(result) > 200 else f"Response: {result}")
            print(f"{'='*60}\n")
            return result

        except openai.APITimeoutError as e:
            elapsed = time.time() - start_time
            timeout_info = getattr(self.client, "timeout", None)
            print(f"[LLM] Timeout ({elapsed:.2f}s) | Timeout setting: {timeout_info if timeout_info is not None else 'N/A'}")
            print("Suggestion: increase timeout or check network connectivity")
            print(f"{'='*60}\n")
            raise RuntimeError(f"LLM API timeout: {e}")

        except openai.APIConnectionError as e:
            base_url = getattr(self.client, "base_url", "N/A")
            print(f"[LLM] Connection error | Base URL: {_sanitize_url_for_logs(base_url)}")
            print("Suggestion: check network connectivity and base_url configuration")
            print(f"{'='*60}\n")
            raise RuntimeError(f"LLM API connection failed: {e}")

        except openai.AuthenticationError as e:
            api_key = getattr(self.client, "api_key", "")
            prefix = (str(api_key)[:6] + "****") if api_key else "N/A"
            print(f"[LLM] Authentication error | API key prefix: {prefix}")
            print("Suggestion: verify API key correctness and expiration")
            print(f"{'='*60}\n")
            raise RuntimeError(f"LLM API authentication failed: {e}")

        except Exception as e:
            elapsed = time.time() - start_time if start_time else 0
            print(f"[LLM] Call failed ({elapsed:.2f}s) | Error: {type(e).__name__}: {e}")
            print(f"{'='*60}\n")
            raise RuntimeError(f"LLM API call failed: {e}")

    def generate(self, prompts: list, stop: Optional[list] = None, **kwargs) -> list:
        """Generate text in batch (compatible with langchain generate)."""
        return [self(prompt, stop=stop, **kwargs) for prompt in prompts]

    @classmethod
    def from_config(cls, config_path: Optional[str] = None, **override_params) -> "OpenAIWrapper":
        """
        Create instance from config file (deprecated; prefer constants.py).

        Note: kept for backward compatibility. Prefer OpenAIWrapper() with constants.py.

        Args:
            config_path: Config file path
            **override_params: Parameters overriding config file values

        Returns:
            OpenAIWrapper instance
        """
        import warnings
        warnings.warn(
            "from_config() is deprecated. Configure constants.py and call OpenAIWrapper() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if config_path is None:
            # If no config file is provided, use constants directly
            return cls(**override_params)
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Read config file
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Extract params from config
        is_azure = "azure_endpoint" in config or "endpoint" in config
        
        if is_azure:
            params = {
                "model_name": config.get("azure_deployment") or config.get("deployment"),
                "openai_api_key": config.get("api_key") or config.get("azure_api_key"),
                "use_azure": True,
                "azure_endpoint": config.get("azure_endpoint") or config.get("endpoint"),
                "azure_deployment": config.get("azure_deployment") or config.get("deployment"),
                "api_version": config.get("api_version"),
                "max_tokens": config.get("max_tokens"),
                "temperature": config.get("temperature"),
                "headers": config.get("headers"),
                "timeout": config.get("timeout"),
                "max_retries": config.get("max_retries"),
            }
        else:
            params = {
                "model_name": config.get("model"),
                "openai_api_key": config.get("openai_api_key") or config.get("api_key"),
                "openai_api_base": config.get("openai_api_base") or config.get("base_url"),
                "max_tokens": config.get("max_tokens"),
                "temperature": config.get("temperature"),
                "headers": config.get("headers"),
                "timeout": config.get("timeout"),
                "max_retries": config.get("max_retries"),
            }
        
        # Apply override params
        params.update(override_params)
        return cls(**params)

    def __repr__(self) -> str:
        """Return object string representation."""
        if self.use_azure:
            return f"OpenAIWrapper(Azure, deployment={self.azure_deployment})"
        else:
            base_url = self._init_kwargs.get("base_url", "default")
            return f"OpenAIWrapper({self.model_name}, base_url={base_url})"
