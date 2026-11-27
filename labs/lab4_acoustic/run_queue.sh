#!/usr/bin/env bash
# Sequential GPU job queue for the lab 4 experiments.
#
#   bash run_queue.sh            # from labs/lab4_acoustic; runs detached from any terminal
#
# Jobs are lines of runs/queue.txt: `name|command` (command is run from labs/lab4_acoustic).
# The file is re-read before every job, so jobs can be appended while the queue runs.
# A job is skipped once runs/<name>.started exists; runs/<name>.done gets its exit code.
# The queue idles while there is nothing to do and exits when runs/STOP appears.
set -u
cd "$(dirname "$0")"
mkdir -p runs
QUEUE=runs/queue.txt
touch "$QUEUE"
while [ ! -e runs/STOP ]; do
    job=""
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        name="${line%%|*}"
        if [ ! -e "runs/$name.started" ]; then job="$line"; break; fi
    done < "$QUEUE"
    if [ -z "$job" ]; then sleep 30; continue; fi
    name="${job%%|*}"
    cmd="${job#*|}"
    date '+%F %T' > "runs/$name.started"
    echo "$(date '+%F %T') START $name: $cmd" >> runs/queue.log
    bash -c "$cmd" > "runs/$name.out" 2>&1
    code=$?
    echo "$code $(date '+%F %T')" > "runs/$name.done"
    echo "$(date '+%F %T') END   $name (exit $code)" >> runs/queue.log
done
echo "$(date '+%F %T') queue stopped" >> runs/queue.log
