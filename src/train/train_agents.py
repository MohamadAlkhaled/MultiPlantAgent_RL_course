from tqdm import trange
from config import EPISODES, MAX_STEPS_PER_EPISODE, ANALYSIS_WINDOW

def train_agents(env_q, env_pg, q_agent, pg_agent, 
                 episodes=EPISODES, 
                 max_steps=MAX_STEPS_PER_EPISODE):

    # Lists to store the total reward for both agents for each episode
    rewards_q, rewards_pg = [], []

    # Dictionary to store the plant height trajectories for early and late stages of training
    growth = {"Q_first": [], "Q_last": [], "PG_first": [], "PG_last": []}

    # Use trange for a progress bar during training
    for ep in trange(episodes, desc="Training Agents"):

       
        # Train q-learning agent
   
        s = env_q.reset()
        # track episode reward and heights
        total_r, ep_ht = 0, [] 
        for _ in range(max_steps):
            a = q_agent.choose_action(s)
            s2, r, d = env_q.step(a)

            # The agent learns from the transitions
            q_agent.learn(s, a, r, s2)
            s, total_r = s2, total_r + r
            ep_ht.append(env_q.heights.copy())
            if d:
                break

        rewards_q.append(total_r)

        # store growth data for the first few episodes
        if ep < ANALYSIS_WINDOW:
            growth["Q_first"].append(ep_ht)
        # Store growth data for the last few episodes    
        if ep >= episodes - ANALYSIS_WINDOW:
            growth["Q_last"].append(ep_ht)

        # Decay epsilon to reduce exploration over time    
        q_agent.decay_epsilon()

        
        # Train Policy Gradient Agent
      
        s = env_pg.reset()
        total_r, ep_ht = 0, []
        for _ in range(max_steps):
            a = pg_agent.choose_action(s)
            s2, r, d = env_pg.step(a)

            # Policy Gradient agent stores rewards for the end-of-episode update
            pg_agent.store_reward(r)

            s, total_r = s2, total_r + r
            ep_ht.append(env_pg.heights.copy())
            if d:
                break

        # At the end of the episode, the Policy Gradient agent learns from all stored transitions
        pg_agent.finish_episode()

        rewards_pg.append(total_r)

        # Store growth data for early and late episodes
        if ep < ANALYSIS_WINDOW:
            growth["PG_first"].append(ep_ht)
        if ep >= episodes - ANALYSIS_WINDOW:
            growth["PG_last"].append(ep_ht)
            
    # return rewards and plant growth data
    return rewards_q, rewards_pg, growth
