"""Separate supervisor; original XY run and code snapshot remain immutable."""
from experiments.xy_qcr_continue import validate_config
from scripts import run_xy_qcr as supervisor


def main():
    supervisor.validate_config=validate_config
    return supervisor.main()


if __name__=='__main__':raise SystemExit(main())
