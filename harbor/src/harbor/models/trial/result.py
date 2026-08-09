import traceback
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from harbor.models.agent.context import AgentContext
from harbor.models.task.config import VerifierEnvironmentMode
from harbor.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from harbor.models.trial.config import TrialConfig
from harbor.models.verifier.result import VerifierResult


class TimingInfo(BaseModel):
    """Timing information for a phase of trial execution."""

    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExceptionInfo(BaseModel):
    """Information about an exception that occurred during trial execution."""

    exception_type: str
    exception_message: str
    exception_traceback: str
    occurred_at: datetime

    @classmethod
    def from_exception(cls, e: BaseException) -> "ExceptionInfo":
        return cls(
            exception_type=type(e).__name__,
            exception_message=str(e),
            exception_traceback=traceback.format_exc(),
            occurred_at=datetime.now(),
        )


class ModelInfo(BaseModel):
    """Information about a model that participated in a trial.

    ``provider`` is optional: when the user runs e.g. ``-m gpt-5.4`` with no
    ``<provider>/`` prefix, the CLI records the model name without a
    provider. Downstream writes to the ``model`` table omit the column so
    the DB default (``'unknown'``) takes over, keeping both sides honest
    about "not specified" vs "explicitly unknown".
    """

    name: str
    provider: str | None = None


class AgentInfo(BaseModel):
    """Information about an agent that participated in a trial."""

    name: str
    version: str
    model_info: ModelInfo | None = None


class StepResult(BaseModel):
    step_name: str
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None


class ContinuousSubmission(BaseModel):
    """One submission verified while the agent was still working.

    Kept per submission rather than only in aggregate: the point of continuous
    verification is the trajectory of attempts, so a run that ends at 0.4 after
    climbing from 0.0 is a different result from one that started there.
    """

    name: str
    index: int
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_sec: float | None = None
    queue_key: str | None = None
    scheduler_wait_sec: float | None = None
    scheduler_acquired_at: datetime | None = None
    verification_started_at: datetime | None = None
    verification_attempts: int = 0
    verification_retry_events: list[dict[str, Any]] = Field(default_factory=list)
    verification_recovery_events: list[dict[str, Any]] = Field(default_factory=list)
    artifact_sha256: str | None = None
    evaluation_fingerprint: str | None = None
    cache_hit: bool = False
    source_evaluation_id: str | None = None
    rewards: dict[str, float | int] | None = Field(
        default=None,
        description=(
            "The verifier's full rewards dict, as it would appear on a "
            "trial-level result. None when the submission was not scored."
        ),
    )
    artifact_path: str | None = Field(
        default=None,
        description=(
            "Path of the archived submission, relative to the trial's "
            "artifacts dir. The bytes that were scored, not the agent's copy, "
            "which it may have since overwritten."
        ),
    )
    result_path: str | None = None
    error: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    retryable_infrastructure_error: bool = False

    @property
    def reward(self) -> float | None:
        """The standard scalar reward, by the same key the job aggregates on."""
        if self.rewards is None:
            return None
        value = self.rewards.get("reward")
        return float(value) if isinstance(value, (int, float)) else None


class ContinuousVerificationSummary(BaseModel):
    """What the agent submitted over the run, and how it scored."""

    submissions: list[ContinuousSubmission] = Field(default_factory=list)

    @property
    def scored(self) -> list[ContinuousSubmission]:
        return [s for s in self.submissions if s.rewards is not None]

    @property
    def best_reward(self) -> float | None:
        return max(
            (s.reward for s in self.scored if s.reward is not None), default=None
        )


class TrialResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task_name: str
    trial_name: str
    trial_uri: str
    task_id: LocalTaskId | GitTaskId | PackageTaskId
    source: str | None = None
    task_checksum: str
    config: TrialConfig
    agent_info: AgentInfo
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    verifier_environment_mode: VerifierEnvironmentMode | None = Field(
        default=None,
        description=(
            "Resolved verifier environment mode ('shared' or 'separate') the "
            "trial-level verify ran in. None for multi-step trials: the mode "
            "is resolved per step there and is not recorded yet."
        ),
    )
    exception_info: ExceptionInfo | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None
    step_results: list[StepResult] | None = None
    continuous_verification: ContinuousVerificationSummary | None = Field(
        default=None,
        description=(
            "Submissions verified during the agent phase, in order. None when "
            "continuous verification was not enabled."
        ),
    )
    verifier_reused_continuous_evaluation_id: str | None = Field(
        default=None,
        description=(
            "Trusted continuous evaluation reused as the final result. None when "
            "the final verifier executed separately."
        ),
    )

    def compute_token_cost_totals(
        self,
    ) -> tuple[int | None, int | None, int | None, float | None]:
        """Sum (n_input_tokens, n_cache_tokens, n_output_tokens, cost_usd).

        Single-step trials record an ``AgentContext`` on ``agent_result``;
        multi-step trials never set that and instead record one per step on
        ``step_results[i].agent_result``. Aggregate whichever is populated.
        Returned fields preserve the same semantics as ``AgentContext`` —
        in particular ``n_input_tokens`` is total input *including* cache.
        """
        if self.agent_result is not None:
            contexts = [self.agent_result]
        elif self.step_results:
            contexts = [
                sr.agent_result
                for sr in self.step_results
                if sr.agent_result is not None
            ]
        else:
            contexts = []

        if not contexts:
            return None, None, None, None

        n_input: int | None = None
        n_cache: int | None = None
        n_output: int | None = None
        cost: float | None = None
        for ctx in contexts:
            if ctx.n_input_tokens is not None:
                n_input = (n_input or 0) + ctx.n_input_tokens
            if ctx.n_cache_tokens is not None:
                n_cache = (n_cache or 0) + ctx.n_cache_tokens
            if ctx.n_output_tokens is not None:
                n_output = (n_output or 0) + ctx.n_output_tokens
            if ctx.cost_usd is not None:
                cost = (cost or 0.0) + ctx.cost_usd

        return n_input, n_cache, n_output, cost
