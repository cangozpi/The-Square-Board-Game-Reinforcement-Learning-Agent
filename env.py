import gym
from gameLogic import GameBoard, Player, Game
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
from collections import deque


class GameEnv(gym.Env):
    def __init__(self, board_size:tuple=(9, 10), player1_agent:Player=None, player2:Player=None, render=False):
        super().__init__()
        assert player1_agent is not None, "player1_agent passed to GameEnv.__init__() must be of type class Player"
        game_board = GameBoard(board_size)
        player1_agent.game_board = game_board # set the game_board of the agent
        if player2 is None:
            player2 = Player(game_board)
        self.game = Game(game_board, player1_agent, player2, render=render)
    
    @staticmethod
    def convert_one_hot_encoded_action_to_edge_coordinate_action(action, board_size):
        board_height = board_size[0]
        board_width = board_size[1]
        if action < (board_width - 1) * board_height: # horizontal edge
            i = action + (action // (board_width - 1))
            action = (i, i+1)
        elif action < (board_height * (board_width - 1)) + ((board_height - 1) * (board_width)): # vertical edge
            row_col_offset = (board_width - 1) * board_height
            i = action - row_col_offset
            action = (i, i+(board_width))
            pass
        else:
            raise Exception(f'action must be in range:[0-{(board_height * (board_width - 1)) + ((board_height - 1) * (board_width))}], but given action:{action}')
        return action
    
    def step(self, action):
        """
        action (tuple): coordinates of the edge to place a wall.
        """
        assert self.game.player_turn == 1, "Can only call step() for player1 (agent) but it is not player1's turn at the moment."
        # convert one-hot encoded action to its corresponding edge coordinate action (tuple)
        action = GameEnv.convert_one_hot_encoded_action_to_edge_coordinate_action(action, self.game.game_board.board_size)

        # take your action on the game_board
        self.game.game_board.place_edge(action)
        # check if the player has created a new square (cycle)
        cycle_coords = self.game.game_board.check_for_cycle(action)
        if cycle_coords:
            self.game.player1.score += len(cycle_coords)
            self.game.player1.owned_square_edge_coords.extend(cycle_coords)
            square_filled_flag = True
        else:
            square_filled_flag = False

        if square_filled_flag == False: # next player's turn
            self.game.player_turn += 1

        self.game.time_count += 1
        self.render()

        done = self.game.is_game_done() # check if player1 has finished the game
        trun = False

        # let the player2 play all his moves
        while self.game.player_turn == 2 and ((not done) and (not trun)): # player2's turn
            square_filled_flag = self.game.player2.play_turn()
            self.render()
            done = self.game.is_game_done() # check if player2 has finished the game
            if square_filled_flag == False : # next player's turn
                self.game.player_turn -= 1
            self.game.time_count += 1
            if done:
                break
        
        if not done:
            assert self.game.player_turn == 1, "it should always be player1_agent's turn after every time env.step() function is called."


        next_state = self.game.game_board.A.copy()
        info = {
            'player1.score': self.game.player1.score,
            'player2.score': self.game.player2.score,
            'player_turn': self.game.player_turn,
            'time_count': self.game.time_count,
            'done': done
        }

        reward = self.calculate_reward(cycle_coords, info)

        return next_state, reward, done, trun, info


    def reset(self, *args, **kwargs):
        # reset game (e.g. game_board, player_turn, player_scores, ...)
        self.game.reset()
        self.render()

        state = self.game.game_board.A.copy()
        info = {
            'player1.score': self.game.player1.score,
            'player2.score': self.game.player2.score,
            'player_turn': self.game.player_turn,
            'time_count': self.game.time_count,
            'done': False
        }

        return state, info

    
    def render(self):
        self.game.draw()
    
    def close(self):
        pass

    def calculate_reward(self, cycle_coords, info):
        # Return number of squares completed by the move as a reward:
        # return len(cycle_coords) 

        # Explicitly reward winning the game and punish loosing it
        num_squares_completed = len(cycle_coords)
        if info['done']:
            if info['player1.score'] > info['player2.score']: # player1 won the game
                winning_reward = 10
                return num_squares_completed + winning_reward
            else: # player1 lost the game
                loosing_reward = -10
                return num_squares_completed + loosing_reward
        else:
            return len(cycle_coords)


def set_seed(seed):
    import torch
    from torch.distributions import Categorical
    # Seed torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(seed))




if __name__ == '__main__':
    # set_seed(42)
    cur_run_path = datetime.now()
    writer = SummaryWriter(log_dir=f'./logs/{cur_run_path}')
    model_save_path = f'model/{cur_run_path}.ckpt'
    model_load_path = f'model/2024-01-08 17:03:02.990584.ckpt'

    board_size = (9, 10)
    is_render = False
    load_model = False
    num_steps = 10000
    save_every_num_epoch = 100
    # HYPERPARAMETERS:
    entropy_coef = 0
    entropy_coef_decay = 0.99
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dim = (board_size[0] * board_size[1]) ** 2
    action_dim = (board_size[0] * (board_size[1] - 1)) + ((board_size[0] - 1) * (board_size[1]))
    lr = 1e-3
    batch_size = 256
    clip_rate = 0.1
    adv_normalization = False
    gamma = 0.99
    lambd = 0.95
    l2_reg = 0
    # ======================================== 

    from PPO import PPO_agent
    ppo_agent = PPO_agent(state_dim, action_dim, lr, device, entropy_coef, entropy_coef_decay, epochs, batch_size, clip_rate, adv_normalization, gamma, lambd, l2_reg)
    player1_agent = Player(game_board=None, ppo_agent=ppo_agent)
    env = GameEnv(board_size=board_size, player1_agent=player1_agent, render=is_render)

    # Training loop
    total_steps = 0
    undiscounted_cum_episode_rewards = []
    cur_episode = 0
    cur_epoch = 0
    player1_win_count_over_last_100_episodes = deque([], maxlen=100)

    # load model from ckpt
    if load_model:
        assert model_load_path != '', 'when load_model=True, model_load_path must be specified'
        ckpt_dict = torch.load(model_load_path)
        player1_agent.ppo_agent.load_state_dict(ckpt_dict['player1_agent.state_dict'])
        total_steps = ckpt_dict['total_steps']
        cur_episode = ckpt_dict['cur_episode']
        cur_epoch = ckpt_dict['cur_epoch']
        player1_win_count_over_last_100_episodes = ckpt_dict['player1_win_count_over_last_100_episodes']
        print(f'successfully loaded from checkpoint: {model_load_path}')


    while True:
        state_buffer = []
        action_buffer = []
        log_prob_action_buffer = []
        reward_buffer = []
        done_buffer = []
        next_state_buffer = []


        state, info = env.reset()
        state = state.reshape(-1) # flatten the adjacency matrix representation
        for step in range(num_steps):
            # choose action
            action, log_prob_action = player1_agent.choose_action(state) # action is the coordinates of the edge to place your block
            # take the action
            next_state, reward, done, trun, info = env.step(action)
            next_state = next_state.reshape(-1) # flatten the adjacency matrix representation

            # record rollout:
            state_buffer.append(state)
            action_buffer.append(action)
            log_prob_action_buffer.append(log_prob_action)
            reward_buffer.append(reward)
            done_buffer.append(done or trun)
            next_state_buffer.append(next_state)

            # logging
            undiscounted_cum_episode_rewards.append(reward)

            if done or trun:
                # print('='*50)
                # print(f'\t\t{"player1(A)" if env.game.player1.score > env.game.player2.score else "player2(B)"} WON!')
                # print(f'GAME END SCORES:\n\tplayer1_score(A): {env.game.player1.score}\n\tplayer2_score(B): {env.game.player2.score}')
                # print('='*50)
                if info['player1.score'] > info['player2.score']: # player1_agent won the episode
                    player1_win_count_over_last_100_episodes.append(1)
                else:
                    player1_win_count_over_last_100_episodes.append(0)
                next_state, info = env.reset()
                next_state = next_state.reshape(-1) # flatten the adjacency matrix representation

                # logging
                writer.add_scalar('train/player1 win count over last 100 episodes vs episode', np.sum(player1_win_count_over_last_100_episodes), global_step=cur_episode)
                writer.add_scalar('train/undiscounted_cumulative_episode_rewards vs episode', np.sum(undiscounted_cum_episode_rewards), global_step=cur_episode)
                undiscounted_cum_episode_rewards = []
                cur_episode += 1

                
            state = next_state

            total_steps += 1
            
        # Train the agent
        print(f'[Entered Training] total_timesteps: {total_steps}')
        actor_epoch_losses, critic_epoch_losses = player1_agent.ppo_agent.train(state_buffer, action_buffer, log_prob_action_buffer, reward_buffer, done_buffer, next_state_buffer)
        print(f'[Exitted Training] total_timesteps: {total_steps}')
        # logging
        for actor_loss, critic_loss in zip(actor_epoch_losses, critic_epoch_losses):
            writer.add_scalar('train/actor_loss vs epoch', actor_loss, global_step=cur_epoch)
            writer.add_scalar('train/critic_loss vs epoch', critic_loss, global_step=cur_epoch)
            # print(f'[epoch:{cur_epoch}] actor_epoch_loss: {actor_loss}, critic_epoch_loss: {critic_loss}')
            cur_epoch += 1

        # save model
        if cur_epoch % save_every_num_epoch == 0:
            print(f'saving model at total_steps: {total_steps}, cur_episode: {cur_episode}, cur_epoch: {cur_epoch}')
            if not os.path.exists('model'):
                os.makedirs(model_save_path)
            torch.save({
                'player1_agent.state_dict': player1_agent.ppo_agent.state_dict(),
                'total_steps': total_steps,
                'cur_episode': cur_episode,
                'cur_epoch': cur_epoch,
                'player1_win_count_over_last_100_episodes': player1_win_count_over_last_100_episodes
            }, model_save_path)