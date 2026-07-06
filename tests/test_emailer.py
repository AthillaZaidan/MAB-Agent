from pathlib import Path

import pytest

from modelwatch.emailer import EmailConfig, email_config_from_env, send_digest_email


class FakeSMTP:
    sent_messages = []

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.started_tls = False
        self.logged_in = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, message):
        self.sent_messages.append((self, message))


def test_send_digest_email_uses_smtp_config(tmp_path, monkeypatch):
    digest = tmp_path / "digest.md"
    digest.write_text("# Digest\n\nBody", encoding="utf-8")
    FakeSMTP.sent_messages = []

    send_digest_email(
        digest,
        EmailConfig(
            to="athillazaidanstudy@gmail.com",
            host="smtp.example.test",
            port=587,
            username="sender@example.test",
            password="secret",
            sender="ModelWatch <sender@example.test>",
        ),
        smtp_cls=FakeSMTP,
    )

    smtp, message = FakeSMTP.sent_messages[0]
    assert smtp.host == "smtp.example.test"
    assert smtp.port == 587
    assert smtp.started_tls is True
    assert smtp.logged_in == ("sender@example.test", "secret")
    assert message["To"] == "athillazaidanstudy@gmail.com"
    assert message["From"] == "ModelWatch <sender@example.test>"
    assert message["Subject"].startswith("ModelWatch Digest")
    assert message.get_content() == "# Digest\n\nBody\n"


def test_email_config_from_env_reads_smtp_settings(monkeypatch):
    monkeypatch.setenv("MODELWATCH_SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("MODELWATCH_SMTP_PORT", "587")
    monkeypatch.setenv("MODELWATCH_SMTP_USERNAME", "sender@gmail.com")
    monkeypatch.setenv("MODELWATCH_SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("MODELWATCH_EMAIL_FROM", "ModelWatch <sender@gmail.com>")

    config = email_config_from_env("athillazaidanstudy@gmail.com")

    assert config.host == "smtp.gmail.com"
    assert config.port == 587
    assert config.username == "sender@gmail.com"
    assert config.password == "app-password"
    assert config.sender == "ModelWatch <sender@gmail.com>"


def test_email_config_from_env_requires_credentials(monkeypatch):
    monkeypatch.delenv("MODELWATCH_SMTP_HOST", raising=False)

    with pytest.raises(RuntimeError, match="MODELWATCH_SMTP_HOST"):
        email_config_from_env("athillazaidanstudy@gmail.com")
