import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


def create_model():
    observation=tf.placeholder(tf.float32,shape=[None,128])
    actions =tf.placeholder(tf.int32,shape=[None])
    rewards =tf.placeholder(tf.float32,shape=[None])

    layer_one=tf.layers.dense(observation,400,activation=tf.nn.relu)
    layer_two=tf.layers.dense(layer_one,200,activation=tf.nn.relu)
    output_layer=tf.layers.dense(layer_two,9)

    predicted_action=tf.multinomial(logits=output_layer,num_samples=1)

    loss_function=tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions,9),logits=output_layer)
    loss=tf.reduce_sum(rewards*loss_function)

    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.99)
    train_operation=optimizer.minimize(loss)
    
    return predicted_action,train_operation,observation,rewards,actions
def run_main():
	predicted_action,train_operation,observations,actions,rewards=create_model()
	mutation_rate=33
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for _ in range(10):
			observations_all=[]
			actions_all=[]
			rewards_all=[]
			done = False
			obs = env.reset()
			while not done:
			    env.render()
			    if(random.randrange(0,99)<mutation_rate):
			    	action = sess.run(predicted_action, feed_dict={observations : np.expand_dims(obs, axis=0)})
			    else:
			    	action = env.action_space.sample()
			    if(action<=9):
			    	obs, reward, done, info = env.step(action)
			    observations_all.append(obs)
			    actions_all.append(action)
			    rewards_all.append(reward)
			sess.run(train_operation,feed_dict={observations : observations_all,actions : actions_all,rewards : np.reshape(rewards_all,(len(rewards_all)))})
env = gym.make('MsPacman-ram-v0')
run_main()