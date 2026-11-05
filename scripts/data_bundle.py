# data_bundle.py
"""
Encrypted copy of the dashboard's data, for the app deployed on Streamlit Community Cloud.

The GitHub repository is public and the Roche data never goes there in clear: this script packs the files the
dashboard reads into one zip, encrypts it (Fernet: AES-128 + HMAC) and writes deploy/data.enc, the only data file
pushed. The key (DATA_KEY) stays in .env on this Mac and in the secrets of the deployed app; without it the file
cannot be read. At its start the deployed app unpacks the files into data/ (unpack_if_needed).

  python scripts/data_bundle.py      rebuild deploy/data.enc after a new run of the pipeline (creates DATA_KEY in
                                     .env the first time), then commit and push deploy/data.enc
"""

import io
import os
import zipfile

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(project_dir, "deploy", "data.enc")
FILES = ["data/cleaned_comments.csv", "data/item_tracking.csv", "data/current_snapshot.csv",
         "data/interim/embedding_cache.npz",   # the language model is only loaded for comments it has never seen
         "data/interim/saved_views.json"]


def build(key):
    from cryptography.fernet import Fernet
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            if os.path.exists(os.path.join(project_dir, name)):
                archive.write(os.path.join(project_dir, name), name)
    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)
    with open(BUNDLE_PATH, "wb") as f:
        f.write(Fernet(key).encrypt(buffer.getvalue()))


def unpack_if_needed(key):
    """Deployed app: write the data files from deploy/data.enc when data/ does not have them all or the bundle is
    newer (a new one was pushed). Returns True when the files were unpacked."""
    paths = [os.path.join(project_dir, name) for name in FILES[:3]]
    if not os.path.exists(BUNDLE_PATH) or (all(map(os.path.exists, paths)) and
                                           os.path.getmtime(paths[0]) >= os.path.getmtime(BUNDLE_PATH)):
        return False
    from cryptography.fernet import Fernet
    with open(BUNDLE_PATH, "rb") as f:
        content = Fernet(key).decrypt(f.read())
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(project_dir)
    return True


if __name__ == "__main__":
    from cryptography.fernet import Fernet
    from dotenv import dotenv_values, set_key
    env_path = os.path.join(project_dir, ".env")
    key = dotenv_values(env_path).get("DATA_KEY")
    if not key:
        key = Fernet.generate_key().decode()
        set_key(env_path, "DATA_KEY", key, quote_mode="never")
        print("New DATA_KEY written to .env: copy it into the secrets of the deployed app.")
    build(key)
    print(f"{os.path.relpath(BUNDLE_PATH, project_dir)}: {os.path.getsize(BUNDLE_PATH) / 1e6:.1f} MB")
