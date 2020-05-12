import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

GAMMA = 0.8  
OBSERVE = 300 
EXPLORE = 100000 
FINAL_EPSILON = 0.0 
INITIAL_EPSILON = 0.8 
REPLAY_MEMORY = 400 
BATCH_SIZE = 64
STEP_SIZE = 5

class BrainDQN:

	def __init__(self,actions,Sensor):

		self.step_size = STEP_SIZE
		self.replayMemory = deque()
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.recording = EXPLORE
		self.sensor_dim = Sensor
		self.actions = actions
		self.hidden1 = 256
		self.hidden2 = 256
		self.hidden3 = 512
		self.createQNetwork()

	def createQNetwork(self):

		#W_fc1 = self.weight_variable([self.sensor_dim,self.hidden1])
		W_fc1 = self.weight_variable([self.hidden1, self.hidden1])
		b_fc1 = self.bias_variable([self.hidden1])

		W_fc2 = self.weight_variable([self.hidden1,self.hidden2])
		b_fc2 = self.bias_variable([self.hidden2])
        
		W_fc3 = self.weight_variable([self.hidden2,self.hidden3])
		b_fc3 = self.bias_variable([self.hidden3])
        
		W_fc4 = self.weight_variable([self.hidden3,self.actions])
		b_fc4 = self.bias_variable([self.actions])        

		self.stateInput = tf.placeholder("float",[None,self.step_size, self.sensor_dim]) #加一个时间步

		#加入LSTM
		lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden1, name='q_lstm')
		lstm_out, state = tf.nn.dynamic_rnn(lstm, self.stateInput, dtype=tf.float32)
		reduced_out = lstm_out[:, -1, :]
		reduced_out = tf.reshape(reduced_out, shape=[-1, self.hidden1])

		h_fc1 = tf.nn.relu(tf.matmul(reduced_out,W_fc1) + b_fc1)
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)
		h_fc3 = tf.nn.tanh(tf.matmul(h_fc2,W_fc3) + b_fc3)        
        
		self.QValue = tf.matmul(h_fc3,W_fc4) + b_fc4

		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
		self.trainStep = tf.train.AdamOptimizer(learning_rate=10**-5).minimize(self.cost)

		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())

	def trainQNetwork(self):
		# minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		# state_batch = [data[0] for data in minibatch]
		# action_batch = [data[1] for data in minibatch]
		# reward_batch = [data[2] for data in minibatch]
		# nextState_batch = [data[3] for data in minibatch]
		idx = np.random.choice(np.arange(len(self.replayMemory) - self.step_size), size=BATCH_SIZE, replace=False)
		res_batch = []
		for i in idx:
			res_temp = []
			for j in range(self.step_size):
				res_temp.append(self.replayMemory[i+j])

			res_batch.append(res_temp)		#获取【batch_size,step_size,4,state_size】

		# state_batch = [data[:,0,:] for data in res_batch]
		state_batch = []
		for i in res_batch:
			state_temp = []
			for step in i:
				temp = step[0]
				state_temp.append(temp)
			state_batch.append(state_temp)
		state_batch = np.asarray(state_batch)

		# print(np.shape(state_batch))

		#nextState_batch = [data[:, 3] for data in res_batch]
		nextState_batch = []
		for i in res_batch:
			state_temp = []
			for step in i:
				temp = step[0]
				state_temp.append(temp)
			nextState_batch.append(state_temp)
		nextState_batch = np.asarray(nextState_batch)

		# print(np.shape(nextState_batch))

		action_batch =  [data[-1][1] for data in res_batch]
		reward_batch = [data[-1][2] for data in res_batch]

		y_batch = []
		# print(np.shape(action_batch))
		# print(np.shape(action_batch))
		QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})
		for i in range(0,BATCH_SIZE):

			y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		_, self.loss = self.session.run([self.trainStep,self.cost],feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})
		return self.loss

	def setPerception(self,nextObservation,action,reward):
		loss = 0
		newState = nextObservation
		self.replayMemory.append((self.currentState,action,reward,newState))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
            
			loss = self.trainQNetwork()

		self.currentState = newState
		self.timeStep += 1
		return loss
        

	def getAction(self):
		if self.timeStep > self.step_size :
			# 修改馈入数据
			temp = []
			for i in range(self.step_size - 1):  # [step_size-1,4]
				temp.append(self.replayMemory[-(1 + i)])
			state_temp = []
			for i in temp:
				temp2 = i[0]
				state_temp.append(temp2)
			state_temp.append(self.currentState)  # 【5，10】
			QValue = self.QValue.eval(feed_dict= {self.stateInput:[state_temp]})
		action = np.zeros(self.actions)
		if  self.timeStep <= self.step_size or random.random() <= self.epsilon:			#这里多加几步初始随机
			action_index = random.randrange(self.actions)
			action[action_index] = 1
		else:
			action_index = np.argmax(QValue)
			action[action_index] = 1
         
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
			self.recording = self.recording-1

		return action,self.recording
    
	def getAction_test(self,observation):
		temp = []
		for i in range(self.step_size-1):		#[step_size-1,4]
			temp.append(self.replayMemory[-(1+i)])
		#print(np.shape(temp))
		state_temp = []
		for i in temp:
			temp2 = i[0]
			state_temp.append(temp2)
		#print(np.shape(state_temp))
		state_temp.append(observation)  #【5，10】

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[state_temp]})		#这里增加一维，【1，5，10】
		action = np.zeros(self.actions)
		action_index = np.argmax(QValue)
		action[action_index] = 1

		return action
    
	def setInitState(self,observation):
		self.currentState = observation

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)
            