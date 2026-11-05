"""Own-session supervisor for the explicitly requested XY dimension control."""
from experiments.xy_qcr_g2 import validate_config
from scripts import run_xy_qcr as supervisor


def main():
    supervisor.validate_config = validate_config
    return supervisor.main()


if __name__ == '__main__':
    raise SystemExit(main())
