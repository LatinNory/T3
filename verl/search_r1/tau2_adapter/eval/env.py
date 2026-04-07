from search_r1.tau2_adapter.core.environment import Environment
from search_r1.tau2_adapter.core.message import AssistantMessage, ToolMessage, UserMessage
from search_r1.tau2_adapter.core.tasks import Task
from search_r1.tau2_adapter.eval.schema import DBCheck, EnvAssertionCheck, RewardInfo


def evaluate_environment(
    environment_constructor,
    task: Task,
    full_trajectory: list[AssistantMessage | UserMessage | ToolMessage],
    solo_mode: bool = True,
    env_kwargs: dict | None = None,
) -> RewardInfo:
    if task.evaluation_criteria is None:
        return RewardInfo(reward=1.0, info={"note": "No evaluation criteria"})

    env_kwargs = env_kwargs or {}
    initialization_data = task.initial_state.initialization_data if task.initial_state else None
    initialization_actions = task.initial_state.initialization_actions if task.initial_state else None
    message_history = task.initial_state.message_history if task.initial_state and task.initial_state.message_history else []

    predicted_environment: Environment = environment_constructor(solo_mode=solo_mode, **env_kwargs)
    predicted_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=full_trajectory,
    )

    gold_environment: Environment = environment_constructor(**env_kwargs)
    gold_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=message_history,
    )

    for action in task.evaluation_criteria.actions or []:
        try:
            gold_environment.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
        except Exception:
            pass

    db_match = (
        gold_environment.get_db_hash() == predicted_environment.get_db_hash()
        and gold_environment.get_user_db_hash() == predicted_environment.get_user_db_hash()
    )
    db_reward = 1.0 if db_match else 0.0
    db_check = DBCheck(db_match=db_match, db_reward=db_reward)

    env_assertion_checks = []
    env_assertion_reward = 1.0
    for env_assertion in task.evaluation_criteria.env_assertions or []:
        success = predicted_environment.run_env_assertion(env_assertion, raise_assertion_error=False)
        env_assertion_checks.append(
            EnvAssertionCheck(
                env_assertion=env_assertion,
                met=success,
                reward=1.0 if success else 0.0,
            )
        )
        env_assertion_reward *= 1.0 if success else 0.0

    reward = db_reward
    reward_breakdown = {"DB": db_reward}
    if task.evaluation_criteria.env_assertions:
        reward *= env_assertion_reward
        reward_breakdown["ENV_ASSERTION"] = env_assertion_reward

    return RewardInfo(
        reward=reward,
        db_check=db_check,
        env_assertions=env_assertion_checks,
        reward_breakdown=reward_breakdown,
    )
