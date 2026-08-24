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
  ctx.on('agent/pre-step', ({ agent }, next) => {
    const current = ctx.goals.get(agent)
    if (current === undefined) {
      ctx.goals.create(agent, { objective })
    } else {
      if (current.objective !== objective) {
        throw new Error('persisted DeepSeek Harness goal does not match this run objective')
      }
      if (current.phase === 'active' && current.activation === 'disarmed') {
        // A supervised CPU relaunch replays the direct human instruction.  That
        // prompt is the authority to re-arm this same persisted goal.
        ctx.goals.resume(agent, { id: current.id, revision: current.revision })
      }
    }
    return next()
  })
}
