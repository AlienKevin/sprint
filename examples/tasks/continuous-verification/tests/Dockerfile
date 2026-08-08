FROM ubuntu:24.04

# The password lives only in the verifier image.  The agent's container has no
# copy of it and no route to this one, so guessing through the submission
# channel is the only way to learn anything about it.
ENV PASSWORD=zephyr

COPY --chmod=755 test.sh /tests/test.sh
