# continuous-verification

An agent that can only learn by asking the grader.

The password exists only in the verifier image. The agent's container has no copy
and no route to it, so the submission channel is the only source of information:
write a guess into `/app/submissions/queue/`, keep working, and read back how many
characters landed. Without `[verifier.continuous]` the task is unsolvable, which
is the point of the example rather than a property you would want in a real task.

`tests/test.sh` grades a continuous submission and the final answer with the same
code, because they are the same question asked at different times. Every guess and
every result is archived under the trial's `artifacts/continuous/`.
