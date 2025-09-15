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
class Joystick_Base(go2_base.Go2Env):
  def __init__(self,...):
      # initialize configure dictionary and add model parameters . These are done in go2.base class
  def reset(self,rng):
      rng, key = jax.random.split(rng) #this is jax's way of creating a random variable key which we use to sample from a distribution
      # randomize attributes such as q0,starting position (xy) ,starting orientation etc

      # we also get a random command :
      
            cmd = jax.random.uniform(
                    key2, shape=(3,), minval=self._cmd_u_min, maxval=self._cmd_u_max
            )
            
      # initialize info dict:
      info = {
      ...
      }
      
      # this is a great way of bookeeping variables and is what jax natively uses in its mjx_env.State class

      # Important: If at any point we want to have a cmd different from this we can just update the state.info["cmd"] value 
      
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
We can inherent from the above and create any environment we want (e.g. change reset,step or rewards).

