import gym
import time
env = gym.make('MsPacman-ram-v0')
observation = env.reset()
print(env.action_space)
for _ in range(10000):
    try:
    	time.sleep(0.01)
    	env.render()
    	action = env.action_space.sample()
    	observation, reward, done, info = env.step(action)
    	print("OBSERVATION : ",len(observation))
    	print("REWARD : ",reward)
    	# print("DONE : ",done)
    	print("INFO : ",info)
    	if done is True:
    		break;
    	if reward !=0:
    		time.sleep(1)
    except:
        env.close()
env.close()
