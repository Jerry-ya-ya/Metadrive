from metadrive import MetaDriveEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config import METADRIVE_CONFIG

from wrappers.turn_slowdown import TurnSlowDownWrapper

def make_metadrive_env(config_override=None):
    config = dict(METADRIVE_CONFIG)

    if config_override:
        config.update(config_override)

    print("num_scenarios =", config.get("num_scenarios"))
    print("start_seed =", config.get("start_seed"))
    print("map =", config.get("map"))

    env = MetaDriveEnv(config)

    env = TurnSlowDownWrapper(
        env,
        max_throttle = 0.6,
        turn_slow_factor = 0.8,
    )

    return env

def build_vec_env(config_override=None):
    def _make_env():
        return make_metadrive_env(config_override)

    env = DummyVecEnv([_make_env])
    env = VecMonitor(env)
    return env

def get_final_status(info):
    if info.get("arrive_dest"):
        return "arrive_dest"
    if info.get("crash"):
        return "crash"
    if info.get("out_of_road"):
        return "out_of_road"
    if info.get("max_step"):
        return "max_step"
    return "unknown"

def print_scoreboard(total_reward, steps, info):
    route_completion = info.get("route_completion", 0.0) * 100
    final_status = get_final_status(info)

    print()
    print("=" * 48)
    print("MetaDrive Evaluation Scoreboard")
    print("=" * 48)
    print(f"Final status     : {final_status}")
    print(f"Route completion : {route_completion:.1f}%")
    print(f"Total steps      : {steps}")
    print(f"Total reward     : {total_reward:.2f}")
    print("=" * 48)