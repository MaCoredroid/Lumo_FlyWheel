from pathlib import Path

TEMPLATE = Path('serving_maintenance/templates/legacy_profile_template.toml')
CONFIG = Path('serving_maintenance/.codex/config.toml')

def main() -> None:
    CONFIG.write_text('# generated from template\n' + TEMPLATE.read_text())

if __name__ == '__main__':
    main()
