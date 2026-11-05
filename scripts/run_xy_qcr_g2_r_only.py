"""Own-session supervisor for the authorized pinned XY terminal R fit."""
from experiments.xy_qcr_g2_r_only import validate_config
from scripts import run_xy_qcr as supervisor


def main():
    supervisor.validate_config = validate_config
    return supervisor.main()


if __name__ == '__main__':
    raise SystemExit(main())
