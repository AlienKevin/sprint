#!/bin/bash
# Score one guess: how many characters are correct and in the right position.
#
# The same script grades a continuous submission and the final answer, because
# they are the same question asked at different times.  Harbor places the file
# at /app/submission/answer.txt either way.
set -u

mkdir -p /logs/verifier
guess=""
[ -f /app/submission/answer.txt ] && guess=$(cat /app/submission/answer.txt)

matched=0
for ((i = 0; i < ${#PASSWORD}; i++)); do
  [ "${guess:i:1}" = "${PASSWORD:i:1}" ] && matched=$((matched + 1))
done

if [ "$guess" = "$PASSWORD" ]; then reward=1; else reward=0; fi

echo "guess=${guess:-<empty>} matched=$matched/${#PASSWORD}"
cat > /logs/verifier/reward.json <<JSON
{"reward": $reward, "matched": $matched, "length": ${#PASSWORD}}
JSON
