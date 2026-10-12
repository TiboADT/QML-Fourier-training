#!/usr/bin/env bash
# Launch run.py (train or frame-potential) in the background, detached from
# the terminal, so a long sweep on a remote machine survives you closing the
# connection (see README's "Remote / background runs" section for
# the reasoning, and the risks worth knowing before you rely on this).
#
# Usage:
#   ./run_background.sh train [run.py train's own flags...]
#   ./run_background.sh frame-potential [run.py frame-potential's own flags...]
#   ./run_background.sh                 # prompts for which one
#   ./run_background.sh --help          # this message
#   ./run_background.sh train --help    # run.py train's own --help (runs in
#                                        # the foreground, not backgrounded --
#                                        # you want to actually read it)
#
# Everything after the subcommand is passed straight through to run.py --
# this script only adds the background-launch wrapper, not argument parsing;
# run.py's own --help is still the source of truth for what's available.
#
# Log and PID files land in results/logs/ (already gitignored via results/),
# named <command>_<timestamp>[.pid] so concurrent or repeated runs never
# overwrite each other's log.
#
# Uses .venv/bin/python3 if present (matching `make install`'s VENV_DIR),
# else plain `python3`; override either with PYTHON=/path/to/python3.

set -euo pipefail

# Goes to the directory this script is in, so it can be run from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# This define a function to show the documentation
usage() {
    sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Check that run.py is present in the same directory as this script
if [[ ! -f run.py ]]; then
    echo "run.py not found in $SCRIPT_DIR -- is this script still next to it?" >&2
    exit 1
fi

# Check for a Python interpreter to use
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="python3"
    [[ -x .venv/bin/python3 ]] && PYTHON=".venv/bin/python3"
fi

# --help anywhere (not just as $1) should show help in the foreground, never
# get backgrounded into a log file you'd have to go find.
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        if [[ "${1:-}" == "train" || "${1:-}" == "frame-potential" ]]; then
            exec "$PYTHON" run.py "$@"
        fi
        usage
        exit 0
    fi
done

# Assure that the commmande given is valide
COMMAND="${1:-}"
if [[ -z "$COMMAND" || "$COMMAND" == -* ]]; then
    # No subcommand given at all, or the first thing we got was a flag
    # (e.g. you ran `./run_background.sh --circuits 1 ...` and forgot
    # train or frame-potential) -- either way, ask rather than guessing, and
    # leave "$@" untouched (there's no subcommand token in it to shift off).
    echo "Which computation do you want to run?"
    select COMMAND in "train" "frame-potential"; do
        [[ -n "$COMMAND" ]] && break
        echo "Please enter 1 or 2."
    done
else
    shift
fi

if [[ "$COMMAND" != "train" && "$COMMAND" != "frame-potential" ]]; then
    echo "Unknown command '$COMMAND' -- expected 'train' or 'frame-potential'." >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

mkdir -p results/logs
STAMP="$(date +%Y_%m_%d__%Hh%M_%S)"
LOG_FILE="results/logs/${COMMAND}_${STAMP}.log"
PID_FILE="results/logs/${COMMAND}_${STAMP}.pid"
EXIT_FILE="results/logs/${COMMAND}_${STAMP}.exitcode"

echo "Launching: $PYTHON run.py $COMMAND $*"
echo "Logging to: $LOG_FILE"

# -u: unbuffered, so the log fills in real time instead of only on flush/exit.
# The exit code is captured *inside* the backgrounded subshell and written to
# EXIT_FILE, rather than relying on `wait "$PID"` after disown below --
# disown doesn't just stop SIGHUP forwarding, it also makes bash unable to
# report that PID's real exit status (confirmed: `false & disown; wait $!`
# reports success regardless of what the child actually did), so `wait`
# can't tell "crashed" from "finished fine" apart once disowned. A file the
# process writes itself has no such problem.
nohup bash -c '"$1" -u run.py "$2" "${@:5}" > "$3" 2>&1; echo $? > "$4"' _ \
    "$PYTHON" "$COMMAND" "$LOG_FILE" "$EXIT_FILE" "$@" &
PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID"

echo "Process started with PID: $PID (also saved to $PID_FILE)"
echo "Process disowned -- you can safely close the terminal."

# A backgrounded+disowned process that fails immediately (bad flag, import
# error, missing file) gives no feedback in the terminal -- you'd only find
# out by checking the log later, possibly much later. Catch that specific
# case here: poll for up to 25s (importing torch+pennylane alone routinely
# takes several seconds -- measured ~8s on my local machine just to reach and
# fail a bad --circuits value, so a short fixed sleep missed it entirely)
# and react the moment it exits, rather than either a fixed sleep too short
# to catch this, or blocking until a genuinely long run finishes.
echo
echo "Waiting up to 25s to see if the process fails immediately (import/flag problem)."
for _ in $(seq 1 25); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
if ! kill -0 "$PID" 2>/dev/null; then
    EXIT_CODE="$(cat "$EXIT_FILE" 2>/dev/null || echo 1)"
    if [[ "$EXIT_CODE" == "0" ]]; then
        echo
        echo "(finished already -- this was a short run, not a startup failure. Log: $LOG_FILE)"
        exit 0
    fi
    echo
    echo "/!\\ process $PID failed on startup (exit code $EXIT_CODE). Log:"
    echo "----------------------------------------------------------------------"
    cat "$LOG_FILE"
    exit 1
fi
echo
echo "----------------------------------------------------------------------------"
echo "Process $PID is still running after 25s -- assuming it started successfully."
echo "----------------------------------------------------------------------------"
echo "Check progress:   tail -f $LOG_FILE"
echo "Check it's alive: kill -0 $PID && echo running"
echo "Stop it:          kill $PID"
