#!/bin/bash
# The oracle knows the password; a real agent has to find it by submitting.
# Submitting first exercises the continuous path, then writes the final answer.
set -u

echo -n "zephyr" > /app/submissions/queue/oracle.txt
for _ in $(seq 1 60); do
  [ -f /app/submissions/results/oracle.txt.json ] && break
  sleep 2
done
cat /app/submissions/results/oracle.txt.json 2>/dev/null

echo -n "zephyr" > /app/submission/answer.txt
