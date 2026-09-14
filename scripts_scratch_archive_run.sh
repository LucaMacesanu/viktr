#!/usr/bin/env bash
set -uo pipefail
cd /scratch/lim2045/icl_ws/viktr/third_party/openpi

CKPT_DIRS=(
  yor_icl_victr_value_expanded
  yor_icl_pi05_expanded_100k
  yor_icl_pi05_expanded_continued_50k
  yor_icl_pi0_fast_expanded_continued_50k
  yor_icl_victr_vision_value_expanded
  pi05_extended_deltarot6d
  pi05_extended_quantilesfixed
  pi05_extended_quantilesfixed_continued_50k
  pi05_extended_quantilesfixed_full_continued_50k
  pi05_extended_quantilesfixed_state_dropout
  viktr_value_50k
  viktr_vision_50k
  yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k
  yor_icl_pi05_expanded_absolute_joint
  yor_icl_pi05_aligned_deltarot6d_sanity15k
  yor_icl_pi05_aligned_q_extended
  yor_icl_pi05_aligned_q_sanity15k
  yor_icl_pi05_canonical_sanity15k
  yor_icl_pi05_canonical_extended
  yor_icl_pi0_fast_canonical
  yor_icl_victr_value_canonical
  yor_icl_victr_vision_canonical
  yor_icl_victr_vision_value_canonical
)

echo "=== START $(date) ==="
for d in "${CKPT_DIRS[@]}"; do
  src="checkpoints/$d"
  dst="/archive/lim2045/openpi_checkpoints/$d"
  if [[ ! -d "$src" ]]; then
    echo "SKIP (missing): $src"; continue
  fi
  if [[ -d "$dst" ]]; then
    echo "SKIP (dest exists): $dst"; continue
  fi
  avail_kb=$(df --output=avail /archive | tail -1)
  need_kb=$(du -sk "$src" | cut -f1)
  if (( need_kb > avail_kb - 20971520 )); then
    echo "ABORT: not enough archive headroom for $d (need ${need_kb}K, avail ${avail_kb}K, keeping 20G safety margin)"
    break
  fi
  echo "MOVING $src -> $dst ..."
  if mv -- "$src" "$dst"; then
    echo "OK: $d"
  else
    echo "FAILED: $d"
  fi
done
echo "=== DONE $(date) ==="
