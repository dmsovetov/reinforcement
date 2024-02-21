import gymnasium
from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def train(env_name):
    env = make_vec_env(env_name, n_envs=16)

    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=True,
        tensorboard_log="./runs/")

    model_name = "ppo-" + env_name
    model.learn(total_timesteps=5000000)
    model.save(model_name)


def test(env_name, filename, publish: bool = False):
    model = PPO.load(filename)
    env = gymnasium.make(env_name, render_mode='human')

    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    #print(mean_reward, std_reward)

    if publish:
        repo_id = 'dmsovetov/ppo-' + env_name

        # TODO: Define the name of the environment
        #env_id = 'LunarLander-v2'

        # Create the evaluation env and set the render_mode="rgb_array"
        eval_env = DummyVecEnv([lambda: Monitor(gymnasium.make(env_name, render_mode="rgb_array"))])

        # TODO: Define the model architecture we used
        model_architecture = "PPO"

        ## TODO: Define the commit message
        commit_message = "Upload PPO LunarLander-v2 trained agent"

        # method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
        package_to_hub(model=model,  # Our trained model
                       model_name=filename.replace('.zip', ''),  # The name of our trained model
                       model_architecture=model_architecture,  # The model architecture we used: in our case PPO
                       env_id=env_name,  # Name of the environment
                       eval_env=eval_env,  # Evaluation Environment
                       repo_id=repo_id,
                       # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/runs
                       commit_message=commit_message,
                       token='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')

    # Enjoy trained agent
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render("human")


if __name__ == '__main__':
    #train('LunarLander-v2')
    test('LunarLander-v2', 'ppo-LunarLander-v2.zip', publish=True)
