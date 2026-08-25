"""Shared limits for public experiment exports."""

# Keep enough recent runs for multi-model comparison batches. The deployment
# bundler still selects only runs referenced by the current batch, so this does
# not cause obsolete artifacts to be published.
PUBLIC_RUN_LIMIT = 64
