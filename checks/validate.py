"""Runs the two_designs calibration checks -- each one confirms a specific
ensemble's frame potential matches its known Haar target, so a failure here
means the estimator (or the ensemble's construction) is broken, not that
you've found something interesting. See the individual check modules
(checks/validate_clifford.py, checks/validate_local_random.py) for what's
actually being tested and why.

    python check.py validate                     # run every check
    python check.py validate --only clifford      # just one, by explicit name
    python check.py validate --only a             # same thing, short alias

Add a new check by: writing checks/validate_something.py with a main(), then
adding one entry to CHECKS below with whatever aliases you want it callable
by (an explicit name is required; short letter aliases are optional but
handy for quick typing).
"""

import argparse

from checks import validate_clifford, validate_local_random

CHECKS = [
    {
        "name": "clifford",
        "aliases": ("a", "clifford"),
        "description": "random Clifford circuits (exact 3-design)",
        "run": validate_clifford.main,
    },
    {
        "name": "local-random",
        "aliases": ("b", "local-random", "local_random"),
        "description": "local random circuits / KAK1 Haar block (brickwork)",
        "run": validate_local_random.main,
    },
]


def _resolve(alias: str):
    for check in CHECKS:
        if alias in check["aliases"]:
            return check
    valid = ", ".join(sorted({a for c in CHECKS for a in c["aliases"]}))
    raise SystemExit(f"Unknown check '{alias}'. Valid: {valid}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices_help = "; ".join(f"{c['name']} (or {'/'.join(c['aliases'])}): {c['description']}" for c in CHECKS)
    p.add_argument("--only", default=None, metavar="CHECK",
                    help=f"run just one check -- {choices_help}. Omit to run all of them.")
    args = p.parse_args(argv)

    to_run = CHECKS if args.only is None else [_resolve(args.only)]

    for check in to_run:
        print(f"=== {check['name']}: {check['description']} ===\n")
        check["run"]()
        print()


if __name__ == "__main__":
    main()
