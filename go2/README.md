### Training Environment Structure

The basic function of an RL env is to have a way to transition from state s to s' with action a and return some reward r.

A basic usage that follows jax syntax:
```bash
from go2.joystick import Joystick
import jax

env = Joystick(task="flat_terrain")
key = jax.random.PRNGKey(0)
state = env.reset(key)

action = env._default_pose
state = env.step(state, action)
```
```bash
class Joystick(go2_base.Go2Env):
  def __init__(self,...):
      # initialize configure dictionary and add model parameters . These are done in go2.base class
  def reset(self,rng):
      rng, key = jax.random.split(rng) #this is jax's way of creating a random variable key which we use to sample from a distribution
      # randomize attributes such as q0,starting position (xy) ,starting orientation etc
      # initialize info dict:
      info = {
        "rng": rng,
        "command": cmd,
        "step":0,
        "steps_until_next_cmd": steps_until_next_cmd,
        "phase":gait.PHASES,
        "phase_dt":2*jp.pi*self.dt*gait_freq,
        "gait_freq":gait_freq,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "H_max":0.1*jp.ones(4),
        "heightscan":heightscan,
        "H_min":0.*jp.ones(4),
        "motor_targets":0.*jp.ones(12),
        "qpos_error_history": qpos_error_history,
        "qvel_history": qvel_history,
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
      }
      
      # this is a great way of bookeeping variables and is what jax natively uses in its mjx_env.State class
      return mjx_env.State(data, obs, reward, done, metrics, info)
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
      #this first part is actually where simulation step is performed
      motor_targets = self._default_pose + action * self._config.action_scale
      data = mjx_env.step(
          self.mjx_model, state.data, motor_targets, self.n_substeps
      )
      
      obs = self._get_obs(data, state.info)
      done = self._get_termination(data)
      
      rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
      )
      rewards = {
          k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
      }
      #here we also update info values 
      state.info["last_last_act"] = state.info["last_act"]
      state.info["last_act"] = action
      state.info["step"] += 1
      phase_unwrapped=state.info["phase"]+state.info["phase_dt"]
      state.info["phase"]=jp.fmod(phase_unwrapped,2*jp.pi)
      state.info["steps_until_next_cmd"] -= 1

      done = done.astype(reward.dtype)
      state = state.replace(data=data, obs=obs, reward=reward, done=done)
      return state
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
      ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
      ...
  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:

    # This gets the observation from the environment. We can either have the same for actor and critic or use an asymetric approach
    # We also give access to the info dict.

    
```
