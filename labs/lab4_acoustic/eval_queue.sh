#!/usr/bin/env bash
# CPU evaluation queue for lab 4: waits for each checkpoint, then runs evaluate.py on it.
#
#   bash eval_queue.sh        # from labs/lab4_acoustic, alongside run_queue.sh
#
# Jobs: `run|step|phones|out` lines of runs/eval_jobs.txt. `step` is a checkpoint step or
# `final` (wait for runs/<run>.done and use the run's best_model). `phones` is gt (the
# shipped eval_sentences.csv) or g2p (data/eval_sentences_g2p.csv). Done jobs leave
# runs/<run>/<out>/mcd.csv and are skipped. Runs on CPU so the GPU keeps training.
set -u
cd "$(dirname "$0")"
PY=../../data/venv_tts/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=runs/eval_queue.log
while [ ! -e runs/STOP_EVAL ]; do
    pending=0
    while IFS='|' read -r run step phones out; do
        case "$run" in ''|'#'*) continue ;; esac
        [ -e "runs/$run/$out/mcd.csv" ] && continue
        pending=1
        if [ "$step" = final ]; then
            [ -e "runs/$run.done" ] || continue
            ckpt=$(ls runs/$run/$run-*/best_model.pth 2>/dev/null | tail -1)
        else
            ckpt=$(ls runs/$run/$run-*/checkpoint_$step.pth 2>/dev/null | tail -1)
        fi
        [ -n "$ckpt" ] || continue
        set_arg="data/eval_sentences.csv"
        [ "$phones" = g2p ] && set_arg="../../data/eval_sentences_g2p.csv"
        echo "$(date '+%F %T') START $run $step $phones -> $out ($ckpt)" >> "$LOG"
        "$PY" evaluate.py --run "runs/$run" --checkpoint "$ckpt" --eval-set "$set_arg" \
            --metadata ../../data/metadata_phones_gt_train.csv --out-name "$out" --save-mels \
            > "runs/$run/$out.log" 2>&1
        echo "$(date '+%F %T') END   $run $out (exit $?)" >> "$LOG"
    done < runs/eval_jobs.txt
    [ "$pending" = 0 ] && break
    sleep 60
done
echo "$(date '+%F %T') eval queue finished" >> "$LOG"
