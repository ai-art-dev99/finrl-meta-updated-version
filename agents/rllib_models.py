# Updated for Ray RLlib 2.48.0
# - Replaces deprecated `ray.rllib.agents.*` imports and Trainer usage
#   with `ray.rllib.algorithms.*` and AlgorithmConfig + .build().
# - Keeps your public API and behavior similar to the original module.
# - A2C/DDPG/TD3 are not in core RLlib anymore; if their contrib packages
#   are installed we use them. Otherwise we fall back to PPO (for A2C)
#   and SAC (for DDPG/TD3) and print a warning.
# - `compute_single_action` is deprecated on the new API stack but still
#   works; see note near Rllib_model for the modern alternative.
#
# Tested with: ray==2.48.0
#
# If you install contrib algos, pip packages are typically:

#   pip install rllib-a2c rllib-ddpg rllib-td3
# and you can then import: from rllib_ddpg.ddpg import DDPGConfig, etc.

import os
import glob
import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

# Optional: add contrib algos if available.
try:
    from rllib_a2c.a2c import A2CConfig  # type: ignore
except Exception:
    A2CConfig = None  # type: ignore

try:
    from rllib_ddpg.ddpg import DDPGConfig  # type: ignore
except Exception:
    DDPGConfig = None  # type: ignore

try:
    from rllib_td3.td3 import TD3Config  # type: ignore
except Exception:
    TD3Config = None  # type: ignore

# "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO
ALGO_CONFIGS = {
    "ppo": PPOConfig,
    "sac": SACConfig,
}

# Add contrib algos if they were successfully imported.
if A2CConfig is not None:
    ALGO_CONFIGS["a2c"] = A2CConfig
if DDPGConfig is not None:
    ALGO_CONFIGS["ddpg"] = DDPGConfig
if TD3Config is not None:
    ALGO_CONFIGS["td3"] = TD3Config


def _resolve_algo_name(name: str):
    """Map deprecated names to supported algorithms when contrib packages are missing."""
    lname = name.lower()
    if lname in ALGO_CONFIGS:
        return lname, None
    # Fallbacks when contrib packages are not installed.
    if lname == "a2c":
        return "ppo", "A2C was removed from core RLlib; using PPO as a close replacement."
    if lname in {"ddpg", "td3"}:
        return "sac", f"{name.upper()} was removed from core RLlib; using SAC as a replacement."
    raise NotImplementedError(f"Algorithm '{name}' is not supported in this build.")


class Rllib_model:
    """Tiny adapter so you can call agent(obs) just like before."""
    def __init__(self, algo):
        self.algo = algo

    def __call__(self, state):
        # NOTE: compute_single_action is deprecated on the new API stack,
        # but still available. For the modern path, use:
        #   module = self.algo.get_module()
        #   out = module.forward_inference({'obs': np.asarray([state])})
        #   action = out['actions'][0]
        return self.algo.compute_single_action(state)


class DRLAgent:
    """
    Drop-in update for Ray 2.48.0 using AlgorithmConfig + .build().
    Keeps your original method names and parameters.
    """

    def __init__(self, env, init_ray: bool = True, **kwargs):
        self.env = env  # env *class* or callable taking env_config and returning an Env.
        self.price_array = kwargs.get('price_array')
        self.tech_array = kwargs.get('tech_array')
        self.turbulence_array = kwargs.get('turbulence_array')
        if init_ray:
            ray.init(ignore_reinit_error=True)

    def _register_env(self, name: str):
        # Always register a creator that calls your env with env_config.
        # Works whether `self.env` is an env class or a function.
        register_env(name, lambda cfg: self.env(cfg))

    def get_model(
        self,
        model_name,
        env_config=PPOConfig(),
        model_config=None,  # Ignored; retained for backward-compat signature
        framework: str = "torch",
    ):
        # Resolve algorithm (with graceful fallbacks for removed algos).
        resolved_name, note = _resolve_algo_name(model_name)

        # Register env under a fixed name; you can pass any env_config at build time.
        self._register_env("finrl_env")

        # Build a fresh AlgorithmConfig using RLlib's typed API.
        cfg_cls = ALGO_CONFIGS[resolved_name]
        cfg = (
            cfg_cls()
            .environment("finrl_env", env_config=env_config)
            .framework(framework)
            .debugging(log_level="WARN")
        )

        if note:
            print(f"[rllib_models] {note}")

        # For backward compatibility, return two values like before.
        # `model` (first return) used to be a module; callers don't actually
        # need it — they only need the config — but we keep the spot.
        return cfg_cls, cfg

    def train_model(
        self,
        model,            # Unused (kept for backward-compat)
        model_name,
        model_config,     # This should be an AlgorithmConfig from get_model()
        total_episodes: int = 100,
    ):
        # Build the Algorithm and train.
        algo = model_config.build()

        # Save checkpoints under ./test_<algo>
        cwd = f"./test_{model_name}"
        os.makedirs(cwd, exist_ok=True)

        for _ in range(total_episodes):
            algo.train()
            # Save a checkpoint after each iteration.
            # Returns a directory path for this checkpoint.
            algo.save_to_path(cwd)

        ray.shutdown()
        return algo

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        env_config,
        agent_path: str,
        init_ray: bool = True,
        model_config=None,  # Ignored; retained for backward-compat signature
        framework: str = "torch",
    ):
        if init_ray:
            ray.init(ignore_reinit_error=True)

        resolved_name, note = _resolve_algo_name(model_name)

        # Register env and build an Algorithm with matching env settings.
        register_env("finrl_env", lambda cfg: env(cfg))
        cfg_cls = ALGO_CONFIGS[resolved_name]
        cfg = (
            cfg_cls()
            .environment("finrl_env", env_config=env_config)
            .framework(framework)
            .debugging(log_level="WARN")
        )
        algo = cfg.build()

        try:
            # New API: restore_from_path expects a directory path returned by save_to_path().
            algo.restore_from_path(agent_path)
            print("Restored from checkpoint:", agent_path)
        except Exception as e:
            raise ValueError(f"Fail to load agent! {e}")

        agent = Rllib_model(algo)
        return agent


class DRLAgent_old:
    """
    Legacy helper kept for parity with your original module.
    Updated to use AlgorithmConfig + .build() under the hood.
    """

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name):
        resolved_name, note = _resolve_algo_name(model_name)
        if note:
            print(f"[rllib_models] {note}")

        cfg_cls = ALGO_CONFIGS[resolved_name]

        # In the "old" path we used to pass the env instance directly.
        # Here we mirror that by registering an env creator that ignores cfg
        # and returns your provided env instance.
        register_env("finrl_env_old", lambda cfg: self.env)

        env_cfg = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }

        cfg = (
            cfg_cls()
            .environment("finrl_env_old", env_config=env_cfg)
            .debugging(log_level="WARN")
        )
        return cfg_cls, cfg

    def train_model(
        self,
        model,            # Unused (kept for backward-compat)
        model_name,
        model_config,     # AlgorithmConfig
        total_episodes: int = 100,
        init_ray: bool = True,
    ):
        if init_ray:
            ray.init(ignore_reinit_error=True)

        algo = model_config.build()
        for _ in range(total_episodes):
            algo.train()

        # Save once at the end (mirrors prior behavior)
        cwd = f"./test_{model_name}"
        os.makedirs(cwd, exist_ok=True)
        algo.save_to_path(cwd)

        ray.shutdown()
        return algo

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        agent_path: str,
    ):
        resolved_name, note = _resolve_algo_name(model_name)
        if note:
            print(f"[rllib_models] {note}")

        cfg_cls = ALGO_CONFIGS[resolved_name]
        cfg = cfg_cls().environment(env).debugging(log_level="WARN")
        algo = cfg.build()

        try:
            algo.restore_from_path(agent_path)
            print("Restored from checkpoint:", agent_path)
        except Exception as e:
            raise ValueError(f"Fail to load agent! {e}")

        # Test (single episode) on the provided env instance.
        state = env.reset()
        episode_returns = []
        episode_total_assets = [getattr(env, "initial_total_asset", 0)]
        done = False
        while not done:
            action = algo.compute_single_action(state)
            state, reward, done, _ = env.step(action)
            if hasattr(env, "amount") and hasattr(env, "price_ary") and hasattr(env, "day") and hasattr(env, "stocks"):
                total_asset = env.amount + (env.price_ary[env.day] * env.stocks).sum()
                episode_total_assets.append(total_asset)
                if hasattr(env, "initial_total_asset") and env.initial_total_asset:
                    episode_returns.append(total_asset / env.initial_total_asset)

        ray.shutdown()
        if episode_returns:
            print("episode return:", episode_returns[-1])
        print("Test Finished!")
        return episode_total_assets
