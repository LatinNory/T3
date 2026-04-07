from search_r1.tau2_adapter.eval.action import evaluate_actions
from search_r1.tau2_adapter.eval.env import evaluate_environment
from search_r1.tau2_adapter.eval.schema import RewardInfo


def evaluate_task(
    task,
    full_trajectory,
    environment_constructor,
    reward_mode: str = "action",
    solo_mode: bool = True,
    env_kwargs: dict | None = None,
) -> RewardInfo:
    reward_mode = reward_mode.lower()
    if reward_mode not in {"action", "env", "action_env"}:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")

    action_reward = evaluate_actions(task, full_trajectory)
    env_reward = evaluate_environment(
        environment_constructor=environment_constructor,
        task=task,
        full_trajectory=full_trajectory,
        solo_mode=solo_mode,
        env_kwargs=env_kwargs,
    )

    if reward_mode == "action":
        return action_reward
    if reward_mode == "env":
        return env_reward

    reward = action_reward.reward * env_reward.reward
    reward_breakdown = {}
    reward_breakdown.update(action_reward.reward_breakdown)
    reward_breakdown.update(env_reward.reward_breakdown)
    return RewardInfo(
        reward=reward,
        action_checks=action_reward.action_checks,
        db_check=env_reward.db_check,
        env_assertions=env_reward.env_assertions,
        reward_breakdown=reward_breakdown,
        info={
            "action": action_reward.info,
            "env": env_reward.info,
        },
    )
