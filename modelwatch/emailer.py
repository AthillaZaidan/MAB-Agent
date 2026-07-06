from __future__ import annotations

import smtplib
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
