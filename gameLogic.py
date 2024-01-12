import numpy as np
from PPO import PPO_agent


class GameBoard:
    def __init__(self, board_size):
        self.board_size = board_size
        self.A = np.zeros((board_size[0] * board_size[1], board_size[0] * board_size[1]), dtype=np.int16) # Adjacency matrix which represents the board
    
    def draw_ascii_board(self, player1_owned_square_edge_coords, player2_owned_square_edge_coords):
        # print(self.A)
        ascii_board_rows = f''
        ascii_board_cols = f''
        for i in range(self.board_size[0] * self.board_size[1]):
            if i + self.board_size[1] < self.board_size[0] * self.board_size[1]: # bottom edge
                ascii_board_cols += f'{"|" if self.is_edge_placed((i, i+self.board_size[1])) else " "}'
                if i + 1 < (((i // self.board_size[1]) + 1) * self.board_size[1]): # right edge
                    ascii_board_cols += " " # middle cell
                else:
                    ascii_board_cols += '\n'
            if i + 1 < (((i // self.board_size[1]) + 1) * self.board_size[1]): # right edge
                ascii_board_rows += f'*{"-" if self.is_edge_placed((i, i+1)) else " "}'
            else: # end of row
                ascii_board_rows += '*\n'

        

        # mark squares owned by players
        for coord_list in player1_owned_square_edge_coords:
            top_edge = sorted(coord_list, key=lambda c: c[0]+c[1])[0]
            bottom_edge = sorted(coord_list, key=lambda c: c[0]+c[1])[-1]
            # print(f'top_edge: {top_edge}, bottom_edge: {bottom_edge}, coord_list: {coord_list}')
            temp_cols = list(ascii_board_cols)
            temp_cols[sum(top_edge)] = 'A'
            ascii_board_cols = ''.join(temp_cols)

        for coord_list in player2_owned_square_edge_coords:
            top_edge = sorted(coord_list, key=lambda c: c[0]+c[1])[0]
            bottom_edge = sorted(coord_list, key=lambda c: c[0]+c[1])[-1]
            # print(f'top_edge: {top_edge}, bottom_edge: {bottom_edge}, coord_list: {coord_list}')
            temp_cols = list(ascii_board_cols)
            temp_cols[sum(top_edge)] = 'B'
            ascii_board_cols = ''.join(temp_cols)

        ascii_board = f''
        for i in range(self.board_size[0] * self.board_size[1]):
            row = ascii_board_rows[(i*self.board_size[1]*2):((i*self.board_size[1]*2)+(self.board_size[1]*2))]
            col = ascii_board_cols[(i*self.board_size[1]*2):((i*self.board_size[1]*2)+(self.board_size[1]*2))]
            ascii_board += row
            if i < self.board_size[1] - 1:
                ascii_board += col

        print(f'ascii_board:\n{ascii_board}')
        # print(f'player1_owned_square_edge_coords: {player1_owned_square_edge_coords}')
        # print(f'player2_owned_square_edge_coords: {player2_owned_square_edge_coords}')
    

    def place_edge(self, coord):
        i, j = coord
        if self.is_edge_placed(coord):
            raise Exception(f'Cannot place an edge which has already been placed (edge_coord: {coord})')
        self.A[i, j] = 1
        self.A[j, i] = 1
        # self.game.draw()
    
    def get_adjacent_nodes(self, coord=()):
        """
        returns coordinates of orthogonally adjacent nodes to the given coord as a list of tuples.
        """
        x, y = coord
        adj_x = list(filter( lambda e: e >= 0,[x-1, x+1]))
        adj_y = list(filter( lambda e: e >= 0, [y-1, y+1]))
        adj_coords = []
        adj_coords += [(e_x, y) for e_x in adj_x]
        adj_coords += [(x, e_y) for e_y in adj_y]
        # print(f'coord: {coord}, adj_cooards: {adj_coords}')
        return adj_coords
    
    def get_horizontal_edge_square_coords(self, coord):
        i, j = coord
        # upper square 
        upper_sqr_coords = []
        if (i - self.board_size[1]) >= 0: # upper square can exist
            upper_sqr_coords = [(i, j), (i - self.board_size[1], i), (i - self.board_size[1], i - self.board_size[1] + 1), (i - self.board_size[1] + 1, j)]

        # bottom square
        bottom_sqr_coords = []
        if (i + self.board_size[1]) < self.board_size[0] * self.board_size[1]: # bottom square can exist
            bottom_sqr_coords = [(i, j), (j, j + self.board_size[1]), (i + self.board_size[1], j + self.board_size[1]), (i, i + self.board_size[1])]
        
        return upper_sqr_coords, bottom_sqr_coords
    

    def get_vertical_edge_square_coords(self, coord):
        i, j = coord
        # left square 
        left_sqr_coords = []
        if (i % self.board_size[1]) > 0: # left square can exist
            left_sqr_coords = [(i, j), (i-1, i), (i-1, j-1), (j-1, j)]

        # right square
        right_sqr_coords = []
        if (((i // self.board_size[1]) + 1) * self.board_size[1]) - i > 1: # right square can exist
            right_sqr_coords = [(i, j), (i, i+1), (i+1, j+1), (j, j+1)]
        
        return left_sqr_coords, right_sqr_coords


    def is_edge_placed(self, coord):
        i, j = coord
        return self.A[i, j] == 1


    def check_for_cycle(self, coord=()):
        """
        check if the given edje (coords) forms and squaes (cycles).
        """
        # Example:
        # vertical edge coord = (5,6) --> [(5,9), (6,10), (9,10), (5,6)] or [(1,2), (2,6), (1,5) (5,6)]
        # horizontal edge coord = (5,9) --> [(4,5), (4,8) (8,9), (5,9)] or [(5,6), (6,10), (9,10), (5,9)]
        i, j = coord
        assert j > i, f'edge coord must have format (i,j) where j > i, but given edge coord: {coord}'

        cycle_coords = []

        if j == i + 1: # horizontal edge coord
            upper_sqr_coords, bottom_sqr_coords = self.get_horizontal_edge_square_coords(coord)
            # print(f'coord: {coord}, upper_sqr_coords: {upper_sqr_coords}, bottom_sqr_coords: {bottom_sqr_coords}')
            upper_count = 0
            for c in upper_sqr_coords:
                if self.is_edge_placed(c):
                    upper_count += 1
                else:
                    break

            bottom_count = 0
            for c in bottom_sqr_coords:
                if self.is_edge_placed(c):
                    bottom_count += 1
                else:
                    break
            
            if upper_count == 4:
                cycle_coords.append(upper_sqr_coords)
                # print(f'\n\tUpper square with edge coords: {upper_sqr_coords} exists')
            if bottom_count == 4:
                cycle_coords.append(bottom_sqr_coords)
                # print(f'\n\tBottom square with edge coords: {bottom_sqr_coords} exists')

        elif j == i + self.board_size[1]: # vertical edge coord
            left_sqr_coords, right_sqr_coords = self.get_vertical_edge_square_coords(coord)
            # print(f'coord: {coord}, left_sqr_coords: {left_sqr_coords}, right_sqr_coords: {right_sqr_coords}')
            left_count = 0
            for c in left_sqr_coords:
                if self.is_edge_placed(c):
                    left_count += 1
                else:
                    break

            right_count = 0
            for c in right_sqr_coords:
                if self.is_edge_placed(c):
                    right_count += 1
                else:
                    break
            
            if left_count == 4:
                cycle_coords.append(left_sqr_coords)
                # print(f'\n\tLeft square with edge coords: {left_sqr_coords} exists')
            if right_count == 4:
                cycle_coords.append(right_sqr_coords)
                # print(f'\n\tRight square with edge coords: {right_sqr_coords} exists')
        else:
            raise Exception(f'Not a valid edge: {coord}')
        
        return cycle_coords
    
    def reset(self):
        self.A = np.zeros((self.board_size[0] * self.board_size[1], self.board_size[0] * self.board_size[1]), dtype=np.int16) # Adjacency matrix which represents the board
        # self.game.draw()



class Player:
    def __init__(self, game_board, ppo_agent:PPO_agent=None):
        self.score = 0 # +1 for every square owned by the player
        self.game_board = game_board
        self.owned_square_edge_coords = []
        self.ppo_agent = ppo_agent
        if self.ppo_agent is not None:
            assert isinstance(self.ppo_agent, PPO_agent), 'ppo_agent passed into <class> Player.__init__(...) must be of type <class> PPO_agent'
        
    def choose_action(self, state=None):
        if self.ppo_agent is not None: # PPO Agent chooses action
            while True: # keep sampling actions from ppo model until a playable action (not played yet action) is found
                action, pi = self.ppo_agent.choose_action(state, deterministic=False)
                from env import GameEnv
                action_coord = GameEnv.convert_one_hot_encoded_action_to_edge_coordinate_action(action, self.game_board.board_size)
                if self.game_board.is_edge_placed(action_coord) == False: # a playable action is found
                    break
                
            return action, pi

        else: # randomly pick an edge which has not been played yet
            possible_edges = []
            for i in range(self.game_board.board_size[0] * self.game_board.board_size[1]):
                if (i - 1) % 1 > 0: # left edge exists
                    possible_edges.append((i-1, i))
                if (((i // self.game_board.board_size[1]) + 1) * self.game_board.board_size[1] ) - i > 1: # right edge exists
                    possible_edges.append((i, i+1))
                if (i - self.game_board.board_size[1]) > 0: # upper edge exists
                    possible_edges.append((i-self.game_board.board_size[1], i))
                if (i + self.game_board.board_size[1]) < self.game_board.board_size[0] * self.game_board.board_size[1]: # bottom edge exists
                    possible_edges.append((i, i+self.game_board.board_size[1]))
            # print(f'possible_edges: {possible_edges}, len: {len(possible_edges)}')
              
            flag = True
            while flag:
                idx = np.random.randint(low=0, high=len(possible_edges))
                i, j = possible_edges[idx]
                coord = (i, j) if i < j else (j, i)
                if self.game_board.is_edge_placed(coord) == False:
                    flag = False
                    break
            # print(f'[Player.choose_action] coord: {coord}, game_board.is_edge_placed: {self.game_board.is_edge_placed(coord)}')

            return coord
    
    def play_turn(self):
        """
        chooses an action and plays a turn on the game_board. Returns True if a square was created with the taken action, else returns False.
        """
        # choose action
        coord = self.choose_action()
        # take your action on the game_board
        self.game_board.place_edge(coord)
        # check if you have created a new square (cycle)
        cycle_coords = self.game_board.check_for_cycle(coord)
        if cycle_coords:
            self.score += len(cycle_coords)
            self.owned_square_edge_coords.extend(cycle_coords)
            return True
        else:
            return False
    
    def reset(self):
        self.score = 0
        # self.game_board.reset()
        self.owned_square_edge_coords = []


class Game:
    def __init__(self, game_board:GameBoard, player1:Player, player2:Player, render=False):
        assert player1.game_board == player2.game_board, "Both players must play on the same game_board"

        self.game_board = game_board
        # self.game_board.game = self
        self.player1 = player1
        self.player2 = player2
        self.player_turn = 1 # 1: player1, 2:player2
        self.time_count = 0

        self.is_render = render

    
    def play(self):
        if self.player_turn == 1: # player1's turn
            square_filled_flag = self.player1.play_turn()
            if square_filled_flag == False: # next player's turn
                self.player_turn += 1

        elif self.player_turn == 2: # player2's turn
            square_filled_flag = self.player2.play_turn()
            if square_filled_flag == False : # next player's turn
                self.player_turn -= 1
        self.time_count += 1
    
    def draw(self):
        if self.is_render:
            self.game_board.draw_ascii_board(self.player1.owned_square_edge_coords, self.player2.owned_square_edge_coords)
    
    def is_game_done(self):
        """
        returns True if game is done, else False
        """
        # print(f'player1_score: {self.player1.score}, player2_score: {self.player2.score}, ending: {((self.game_board.board_size[0] - 1) * (self.game_board.board_size[1] - 1))}')
        if self.player1.score + self.player2.score == (((self.game_board.board_size[0] - 1) * (self.game_board.board_size[1] - 1))):
            return True
        else:
            return False
    
    def reset(self):
        self.game_board.reset()
        self.player1.reset()
        self.player2.reset()
        self.player_turn = 1
        self.time_count = 0


if __name__ == '__main__':
    board_size = (4,5) # original board game size is (9,10)

    game_board = GameBoard(board_size)
    player1 = Player(game_board)
    player2 = Player(game_board)
    game = Game(game_board, player1, player2, render=True)
    while True:
        game.draw()
        # player takes an action
        game.play()
        game.draw()
        # check if game is done
        if game.is_game_done():
            print('='*50)
            print(f'GAME FINISHED:\n\tplayer1_score(A): {game.player1.score}\n\tplayer2_score(B): {game.player2.score}')
            print('='*50)
            quit()


    # TODO: log played games as pickled np files
    