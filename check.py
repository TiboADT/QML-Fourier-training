"""
Single CLI entry point for the repo's validation/benchmark scripts, which
live in checks/ (kept separate from the library code in circuits.py,
frame_potential.py, two_designs/, etc. -- checks/ is things you *run*, the
rest is things you *import*).

Examples
--------
    python check.py benchmark                     # timing benchmarks (everything)
    python check.py benchmark --only fp           # timing benchmarks (frame potential only)
    python check.py validate                      # every two_designs calibration check
    python check.py validate --only clifford      # just the Clifford-group one
    python check.py validate --only a             # same thing, short alias
    python check.py validate --only local-random  # just the local-random-circuit one

Run `python check.py validate --only <anything>` with a bad value, or
`python check.py validate -h`, to see the full list of valid checks and
their aliases -- that list lives in checks/validate.py, not here.
"""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["benchmark", "validate"])
    p.add_argument("rest", nargs=argparse.REMAINDER,
                    help="remaining arguments, passed through to the chosen command's own CLI")
    args = p.parse_args()

    if args.command == "benchmark":
        from checks.benchmark import main as benchmark_main
        benchmark_main(args.rest)
    elif args.command == "validate":
        from checks.validate import main as validate_main
        validate_main(args.rest)


if __name__ == "__main__":
    main()
