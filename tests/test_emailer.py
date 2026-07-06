from pathlib import Path

from modelwatch.emailer import EmailConfig, send_digest_email


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
