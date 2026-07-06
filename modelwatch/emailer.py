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


def email_config_from_env(to: str) -> EmailConfig:
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
