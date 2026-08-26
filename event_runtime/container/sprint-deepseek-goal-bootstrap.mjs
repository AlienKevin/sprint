/**
 * Create DeepSeek Harness's native persisted goal before the first model step.
 *
 * The headless JSON-RPC adapter intentionally does not consume human slash
 * commands.  This Loader plugin follows DeepSeek Harness's own seed-goal test
 * pattern: it mutates the native GoalService at `agent/pre-step`, after the
 * direct human prompt has been admitted but before the first model request.
 */

export const name = 'sprint-deepseek-goal-bootstrap'
export const inject = ['goals']

const HOST_OWNED_MUTATIONS = ['edit', 'pause', 'complete', 'clear', 'resume']

function requiredObjective() {
  const objective = process.env.DSH_GOAL_OBJECTIVE?.trim()
  if (!objective) throw new Error('DSH_GOAL_OBJECTIVE must be a non-empty string')
  if (/^\/goal(?:\s|$)/u.test(objective)) {
    throw new Error('DSH_GOAL_OBJECTIVE must not contain a textual /goal command')
  }
  return objective
}

export function apply(ctx) {
  const objective = requiredObjective()

  // A benchmark rollout is one host-owned goal whose lifetime is the cost
  // budget.  The stock model-facing goal tools are useful for inspection, but
  // their direct-human authority normally lasts for the whole initial turn.
  // That lets the model edit, pause, or complete the seeded goal even though
  // the benchmark instruction explicitly says to keep working until the host
  // stops it.  An edit used to mutate durable state successfully and then make
  // the invariant below crash the harness on the following pre-step.
  //
  // Put the authority boundary at the service itself so every caller (current
  // tools and future plugins alike) receives an ordinary tool error before any
  // goal/change event is committed. ``block`` remains available to the trusted
  // stock round driver so queue, checkpoint, and round-limit failures become a
  // durable terminal goal state. No model-facing goal tools are mounted.
  for (const mutation of HOST_OWNED_MUTATIONS) {
    if (typeof ctx.goals[mutation] !== 'function') {
      throw new Error(`DeepSeek Harness goal service is missing ${mutation}()`)
    }
    ctx.goals[mutation] = () => {
      throw new Error(
        `benchmark goal is host-owned; ${mutation} is disabled until the budget controller stops the run`,
      )
    }
  }

  ctx.on('agent/pre-step', ({ agent }, next) => {
    const current = ctx.goals.get(agent)
    if (current === undefined) {
      ctx.goals.create(agent, { objective })
    } else if (current.objective !== objective) {
      throw new Error('persisted DeepSeek Harness goal does not match this run objective')
    }
    return next()
  })
}
