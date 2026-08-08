# Sprint evaluation resource isolation

## Enforced budget

Each model/run has an independent resource and queue namespace:

- exactly one supervised CPU agent sandbox (`cpus=4` physical cores,
  `gpus=0`, 16 GiB);
- at most one active agent-controlled A10G training worker, enforced by the
  host dispatcher across polling cycles and retries;
- queued training jobs wait behind that worker instead of allocating another;
- one trusted A10G verifier slot for that run's scoring queue; and
- a separate durable Modal Volume, Harbor trial directory, scoring queue key,
  checkpoint namespace, lease namespace, and timeline namespace.

The verifier is benchmark infrastructure, not compute the agent can execute
inside. It may overlap a training worker, so the platform can briefly have two
GPUs attributed to a run: one agent-controlled training GPU and one sealed
scoring GPU. If the desired accounting rule is one GPU total including trusted
scoring, training and verification need a shared cross-service semaphore; that
is not the current contract.

## Credential boundary

The launcher builds the agent environment from an allowlist and refuses Modal,
AWS, Google Cloud, or Azure control-plane credential names. Modal credentials
remain only on the trusted host/controller. They are not passed through Harbor
agent environment arguments or the per-run agent env file.

## Network boundary

Evaluation runtime is deny-by-default:

- the CPU agent sandbox starts with `network_mode = "no-network"`;
- the trusted launcher derives one audited model API hostname from the selected
  provider and exposes only that hostname during `agent.run()`;
- Codex and Claude Code are pinned and baked into the image, so agent setup does
  not need npm, GitHub, Node, or Claude download access;
- both per-job and standing A10G training workers use Modal
  `block_network=True`; and
- the sealed verifier uses `network_mode = "no-network"`.

The active model hosts are `api.openai.com`, `api.anthropic.com`,
`api.deepseek.com`, and `openrouter.ai`. A run receives only its selected host,
not the whole set. Custom endpoint hosts fail launcher validation until they are
reviewed and added explicitly. Modal API and workspace domains are never in an
evaluation allowlist.

The agent still receives its model-provider credential because Codex or Claude
Code currently runs inside the same sandbox as agent-authored commands. This is
not an adversarially secure model-call boundary: a process with the same user
can potentially reuse that credential against the single allowed model host.
Prompt instructions and normal CLI budgets do not prevent it.

Before treating intentionally hostile agents as isolated, put model inference
behind a per-run gateway token that:

1. is valid only for the assigned model and endpoint;
2. enforces the run's request, token, cost, and wall-clock budgets;
3. records every request in the unified timeline;
4. cannot call cloud control-plane APIs; and
5. replaces the provider hostname in the allowlist so the gateway is the only
   inference route.

Until that gateway and egress rule exist, the infrastructure enforces CPU/GPU
allocation fairness but does not cryptographically prevent unmetered model-API
reuse.
