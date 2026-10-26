#!/bin/bash
cd "$HOME/strata-dev"
Q="$HOME/Strata/strata-nvfp4-nvfp4.json"
D="$HOME/Strata/strata-deepseek-v4.1-flash-ist-das-1.75bpw.json"
EN="--arg=--mtp-draft-vocab --arg=$HOME/Strata/data/draft_vocab_en.bin"
for r in 1 2; do
  python3 exp.py cq_before_$r --base $Q --bench coding --rounds 1 --env STRATA_HC_Q8=0 --env STRATA_PLE_BATCH=0 --drop-mtp-q4 | grep -E "RESULT|NOT READY" | sed "s/^/qwen before /"
  python3 exp.py cq_after_$r  --base $Q --bench coding --rounds 1 | grep -E "RESULT|NOT READY" | sed "s/^/qwen after /"
  python3 exp.py cq_aften_$r  --base $Q --bench coding --rounds 1 $EN | grep -E "RESULT|NOT READY" | sed "s/^/qwen after+en /"
done
for r in 1 2; do
  python3 exp.py cd_before_$r --base $D --bench coding --rounds 1 --arg=--dspark-conf --arg=0.65 | grep -E "RESULT|NOT READY" | sed "s/^/deepseek before /"
  python3 exp.py cd_after_$r  --base $D --bench coding --rounds 1 | grep -E "RESULT|NOT READY" | sed "s/^/deepseek after /"
done
