from dog_walker.config import load_config


def test_load_config_reads_all_sections(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[provider]\n'
        'name = "anthropic"\n'
        'model = "claude-haiku-4-5-20251001"\n'
        'max_tokens = 4096\n'
        '[harness]\n'
        'max_iterations = 20\n'
        'confirm_bash = true\n'
        '[tools]\n'
        'enabled = ["read_file", "run_bash"]\n'
        '[storage]\n'
        'backend = "sqlite"\n'
        'path = "data.db"\n'
    )
    cfg = load_config(str(cfg_file))
    assert cfg.provider_name == "anthropic"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.max_tokens == 4096
    assert cfg.max_iterations == 20
    assert cfg.confirm_bash is True
    assert cfg.enabled_tools == ["read_file", "run_bash"]
    assert cfg.storage_backend == "sqlite"
    assert cfg.storage_path == "data.db"
