import json

from modelwatch.config import load_config


def test_load_config_reads_vector_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "vector_database_path": "data/vectors.sqlite",
                "ollama_embedding_model": "nomic-embed-text",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.vector_database_path.as_posix() == "data/vectors.sqlite"
    assert config.ollama_embedding_model == "nomic-embed-text"


def test_load_config_reads_email_recipient(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"email_to": "athillazaidanstudy@gmail.com"}), encoding="utf-8")

    assert load_config(path).email_to == "athillazaidanstudy@gmail.com"
