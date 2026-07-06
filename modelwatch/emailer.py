from __future__ import annotations

import smtplib
import os
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path


@dataclass
class EmailConfig:
    to: str
    host: str
    port: int
    username: str
    password: str
    sender: str


def _load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def email_config_from_env(to: str) -> EmailConfig:
    _load_dotenv()
    required = [
        "MODELWATCH_SMTP_HOST",
        "MODELWATCH_SMTP_USERNAME",
        "MODELWATCH_SMTP_PASSWORD",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing email environment variable: {missing[0]}")
    username = os.environ["MODELWATCH_SMTP_USERNAME"]
    return EmailConfig(
        to=to,
        host=os.environ["MODELWATCH_SMTP_HOST"],
        port=int(os.environ.get("MODELWATCH_SMTP_PORT", "587")),
        username=username,
        password=os.environ["MODELWATCH_SMTP_PASSWORD"],
        sender=os.environ.get("MODELWATCH_EMAIL_FROM", username),
    )


def send_digest_email(digest_path: str | Path, config: EmailConfig, smtp_cls=smtplib.SMTP) -> None:
    path = Path(digest_path)
    message = EmailMessage()
    message["Subject"] = f"ModelWatch Digest {path.stem.removeprefix('digest-')}"
    message["From"] = config.sender
    message["To"] = config.to
    message.set_content(path.read_text(encoding="utf-8"))

    with smtp_cls(config.host, config.port) as smtp:
        smtp.starttls()
        smtp.login(config.username, config.password)
        smtp.send_message(message)
