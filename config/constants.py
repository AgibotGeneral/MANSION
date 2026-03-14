"""Constants and environment configuration for Mansion."""

from __future__ import annotations

import os
from pathlib import Path


ABS_PATH_OF_MANSION = os.path.abspath(os.path.join(os.path.dirname(Path(__file__)), ".."))

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
HD_BASE_VERSION = os.environ.get("HD_BASE_VERSION", "2023_09_23")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser("~/.objathor-assets")
)

OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(
    OBJATHOR_VERSIONED_DIR, "annotations.json.gz"
)

MANSION_BASE_DATA_DIR = os.path.join(
    OBJATHOR_ASSETS_BASE_DIR, "holodeck", HD_BASE_VERSION
)

MANSION_THOR_FEATURES_DIR = os.path.join(
    MANSION_BASE_DATA_DIR, "thor_object_data"
)
MANSION_THOR_ANNOTATIONS_PATH = os.path.join(
    MANSION_THOR_FEATURES_DIR, "annotations.json.gz"
)

THOR_COMMIT_ID = os.environ.get(
    "THOR_COMMIT_ID", "6f165fdaf3cf2d03728f931f39261d14a67414d0"
)

# ============================================================================
# LLM configuration (centralized)
# ============================================================================

# Profile 1: OpenAI-compatible
# Required: OPENAI_API_KEY
# Optional: OPENAI_API_BASE (proxy URL; defaults to official https://api.openai.com/v1)
#           OPENAI_MODEL    (model name; defaults to gemini-2.5-pro)
OPENAI_CONFIG = {
    "provider": "openai",
    "model": os.getenv("OPENAI_MODEL", "gemini-2.5-pro"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_API_BASE"),  
    "max_tokens": 20000,
    "temperature": 0.3,
}

# Profile 2: Azure OpenAI
# Required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
# Optional: AZURE_OPENAI_DEPLOYMENT (deployment name; defaults to gpt-5)
#           AZURE_OPENAI_API_VERSION
AZURE_CONFIG = {
    "provider": "azure",
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
    "model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
    "max_tokens": 20000,
    "temperature": 0.3,
}

# Profile 3: azure_gpt5 — custom Azure deployment for per-node overrides in mixed mode.
# Uses the same credentials as AZURE_CONFIG. Set AZURE_OPENAI_DEPLOYMENT_GPT5 to point
# to a different deployment (e.g. a GPT-5 or other high-performance model).
# This profile is not a top-level llm_provider option; it is referenced by
# node_config.json entries in mixed mode.
AZURE_GPT5_CONFIG = {
    "provider": "azure",
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")),
    "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
}

# Default global config (backward compatibility)
LLM_PROVIDER = "openai"
OPENAI_API_KEY = OPENAI_CONFIG["api_key"]
AZURE_OPENAI_API_KEY = AZURE_CONFIG["api_key"]
AZURE_GPT5_API_KEY = AZURE_GPT5_CONFIG["api_key"]
LLM_MODEL_NAME = OPENAI_CONFIG["model"]
OPENAI_API_BASE = OPENAI_CONFIG["base_url"]
AZURE_OPENAI_ENDPOINT = AZURE_CONFIG["endpoint"]
AZURE_OPENAI_DEPLOYMENT = AZURE_CONFIG["deployment"]
AZURE_OPENAI_API_VERSION = AZURE_CONFIG["api_version"]
LLM_MAX_TOKENS = 20000
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 200.0
LLM_MAX_RETRIES = 3

DEBUGGING = os.environ.get("DEBUGGING", "0").lower() in ["1", "true", "t"]

LOCAL_AI2THOR_PATH = os.environ.get("LOCAL_AI2THOR_PATH", os.path.expanduser("~/.ai2thor/releases/thor-Linux64-local/thor-Linux64-local"))
