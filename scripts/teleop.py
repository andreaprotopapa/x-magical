import gym
from absl import app, flags

from xmagical import register_envs
from xmagical.utils import KeyboardEnvInteractor
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_name",
    "SweepToTop-Gripper-State-Allo-Demo-v0",
    "The environment to load.",
)
flags.DEFINE_boolean("exit_on_done", False, "Whether to exit if done is True.")


def main(_):
    register_envs()
    env = gym.make(FLAGS.env_name)
    viewer = KeyboardEnvInteractor(action_dim=env.action_space.shape[0])

    env.reset()
    obs = env.render("rgb_array")
    viewer.imshow(obs)

    i = [0]

    def step(action):
        obs, rew, done, info = env.step(action)
        if obs.ndim != 3:
            obs = env.render("rgb_array")
            plt.imshow(obs)
            plt.show(block=False)
            # Access the window manager for the plot window
            plot_window = plt.gcf().canvas.manager.window
            # Set the window position using geometry (x, y position)
            plot_window.move(300, 300)  # Set the plot window at position (300, 300) on the screen
            plt.pause(0.5)
        if done and FLAGS.exit_on_done:
            return
        if i[0] % 100 == 0:
            print(f"Done, score {info['eval_score']:.2f}/1.00")
        i[0] += 1
        return obs

    viewer.run_loop(step)
    plt.close()


if __name__ == "__main__":
    app.run(main)
