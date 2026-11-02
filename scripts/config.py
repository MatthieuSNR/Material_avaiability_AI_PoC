# config.py
# API keys and the app password are read from the .env file at the project root (never pushed to GitHub, see
# .gitignore). Copy .env.example to .env and fill DEEPL_API_KEY, MISTRAL_API_KEY and, to share the app, APP_PASSWORD.
# Deployed on Streamlit Community Cloud (no .env there): the same names are read from the app's secrets.

import os

from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
ON_CLOUD = not os.path.exists(ENV_PATH)   # deployed copy: password required, data from deploy/data.enc
_PASSWORD_FROM_SHELL = "APP_PASSWORD" in os.environ   # e.g. APP_PASSWORD= for a local test without password
load_dotenv(ENV_PATH)
if ON_CLOUD:
    try:
        import streamlit as st
        for name, value in st.secrets.items():   # the modules read os.getenv: secrets as environment variables
            if isinstance(value, str):
                os.environ.setdefault(name, value)
    except Exception:   # no secrets file (e.g. tests): nothing to add
        pass

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")


def app_password():
    """Password of the dashboard: the one given when starting it (APP_PASSWORD=... streamlit run), otherwise the
    one in .env, read again at every call so that a change applies without restarting the dashboard."""
    if _PASSWORD_FROM_SHELL or not os.path.exists(ENV_PATH):
        return os.getenv("APP_PASSWORD", "")
    from dotenv import dotenv_values
    return dotenv_values(ENV_PATH).get("APP_PASSWORD") or ""
