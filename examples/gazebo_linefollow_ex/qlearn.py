import random
import pickle
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        with open(filename+'.pickle', 'rb') as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename+'.pickle', 'wb') as f:
            pickle.dump(self.q, f)
        with open(filename+'.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['State', 'Action', 'Q-value'])
            for state, actions in self.q.items():
                if isinstance(actions, dict):
                    for action, q_val in actions.items():
                        writer.writerow([state, action, q_val])

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        q = [self.getQ(state,a) for a in self.actions]
        num_actions = len(self.actions)
        maxQ = max(q)
    

        #q_values = self.q.get(state, {})

        if random.uniform(0,1) < self.epsilon: #exploration
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - 0.5*mag for i in range(num_actions)] #modifying the matrix by rescaling it 
            maxQ = max(q)
            

            action = random.choice(self.actions)
        else: #exploitation
            maxQ = max(q)
            #actions_with_max_q = [a for a, q in q_values.items() if q == maxQ]
            actions_with_max_q = [a for a in self.actions if self.getQ(state, a) == maxQ]
            if not actions_with_max_q:
                # choose a random action if no actions have a defined Q-value
                action = random.choice(self.actions)
            else:
                # choose a random action from the ones with the highest Q-value
                action = random.choice(actions_with_max_q)

            action = random.choice(actions_with_max_q)

        if return_q: #if they want it, give it
            #q_value = q_values.get(action, 0)
            return action, q #q_value


        return action


    #added learnQ based on video
    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        # Get current Q-value for the (state1, action1) pair, or set it to 0 if it
        # doesn't exist yet.
        q_current = self.q.get((state1, action1), 0)

        # Get the maximum Q-value for the next state (state2) and any possible actions
        # we could take from there.
        q_next_max = max([self.getQ(state2, a) for a in self.actions])

        # Calculate the updated Q-value for the (state1, action1) pair using the
        # Bellman update equation.
        q_updated = q_current + self.alpha * (reward + self.gamma * q_next_max - q_current)

        # Update the Q-value dictionary with the new Q-value for the (state1, action1) pair.
        self.q[(state1, action1)] = q_updated