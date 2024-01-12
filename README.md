The "Square" Board Game (a.k.a "Square MB spiele kluge vorausplanung") Reinforcement Learning Agent
---
This code implements an openAI Gym environment for the _"Square"_ board game and applies reinforcement learning (PPO) to train agents.

---

### Contributions:
* Implements _Square_ game using graph structure and builds an OpenAI gym env on top of it to train RL agents.
* Implements PPO Reinforcement Learning agent to play the game by making it play against another player who takes random actions.
* _TODO:_ implement a GUI to allow one to play against a trained RL agent.
* _TODO:_ implement a rl agent which uses GNNs (graph neural networks).

---

### The "Square" Board Game Rules:
_Note: Refer to http://www.superfred.de/square.html for the original explanation of the game rules described below._

The aim of the game is to capture more boxes on the game board.  
There are 10x9 pegs on the plastic game board; the white sticks can be placed between two horizontally or vertically adjacent pegs. If four connected pegs form a square, the interior space is considered conquered. To indicate this, the last player to contribute a stick to this square places a tile in his player color in the middle of the square.  
At the beginning the game board is completely empty. The players now take turns placing sticks on the playing field in any free spaces. When a player completes a square, he marks it with his tile and can then immediately place another stick. By completing it again, longer chain hoists are possible.
The game ends when all 72 fields have been conquered. The player who now owns the most spaces on the board has won this game.


In other words:  
In the Square board game, player's take turns placing one edge at a time on a grid like board. The placed edges can form 1x1 squares and the one gets to complete the square's last edge during their turn gets to own that square. Upon completing a square on a given turn the same player gets to make another move. This goes on until the player cannot complete yet another square during their turn. At the end of the game the player who owns the most of the squares wins the game.

<img src="https://i.ebayimg.com/images/g/bywAAOSwNGNlepky/s-l1600.jpg" width="1500px" height="1000px">

<img src="https://lh3.googleusercontent.com/proxy/Zenx5WRGod965BLyZ5nT7hOj5m9Cw7ACEf0BXEQsdu0XcuJ-BmJ_koVzj4_9eMqdxQYd808aocXZwA" width="800px" height="500px">

---

### Dependencies
* Installation with conda:
    ```bash
    conda create --name <env> python=3.8.16 --file requirements.txt
    conda activate <env>
    ```
    ```bash
    pip3 install -r requirements.txt
    ```
---
### Running the code:

* To train:
    ```bash
    python3 envs.py
    ```
* To view training logs:
    ```bash
    tensorboard --logdir=./logs
    ```

    __Sample training plots for a game with board_size=(4, 5)__
        <img src="./assets/4x5 game board training logs.png" width="1200px" height="600px">

---

### Setting hyperparameters, saving & loading model configurations:
Navigate to __main__ section in _env.py_ and modify the following parameters to your needs ->
```python
model_save_path = f'model/{cur_run_path}.ckpt' # path the trained model will be saved
model_load_path = f'model/2024-01-08 15:50:41.968535.ckpt' # path the model will be loaded from given

board_size = (9, 10) # dimensions of the board (board_height, board_width)
is_render = False # ascii rendering of the board state on the terminal
load_model = False # if True a trained model will be loaded from the checkpoint specified by the model_load_path parameter
num_steps = 1000 # number of steps to take (gather rollout) everytime before entering training
save_every_num_epoch = 20 # model will be saved after every this many epochs have passed
# HYPERPARAMETERS:
entropy_coef = 0 # entropy coefficient in PPO
entropy_coef_decay = 0.99 # entropy coefficient is decayed this much by multiplying it with this specified value
epochs = 10 # number of epochs to train on every call to train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 1e-4 # learning rate used for both actor and critic networks of PPO
batch_size = 64
clip_rate = 0.1 # PPO clip rate
adv_normalization = False # whether to normalize advantages in PPO
gamma = 0.99
lambd = 0.95
l2_reg = 0 # l2 regularization coefficient in PPO
```

---

### Game Environment:
Game board is though of as a graph structure where pegs on the game board constitute nodes of the graph, and the white sticks placed in between those pegs constitute the edges in the graph. The _state_'s (game board) are represented as an adjacency matrix (https://en.wikipedia.org/wiki/Adjacency_matrix). Actions are tuples of node (pegs) id's which the white stick will be placed in between.  
For example, For for a gameboard with board_size=(4, 5). The board will be represented with nodes = [0, 1, 2, ..., 18, 19] as


_Note: (*'s represent pegs on the board, - and | represent white sticks placed on the board by the players, and A means that the corresponding square is owned by the player1, and B means that the corresponding square is woned by the player2)_

* As ASCII rendering:
    ```text
    * * * * *

    * * * * *

    * * * * *

    * * * * *
    ```
* As graph node id's:
    ```text
    0 1 2 3 4
    5 6 7 8 9
    10 11 12 13 14
    15 16 17 18 19
    ```
Then for example to place a white stick between nodes 14 and 19 we use _action=(14, 19)_ which yields:
* As ASCII rendering after taking the _action=(14,19)_:
    ```text
    * * * * *

    * * * * *

    * * * * *
            |
    * * * * *
    ```

* One can render the game as ASCII text on the terminal by setting _is\_render=True_ in the _env.py_ file. A sample gameplay with _board_size=(3, 4)_ is pasted below:
    ```text
    ascii_board:
    * * * *
       
    * * * *
       
    * * * *

    ascii_board:
    * * * *
       
    * *-* *
       
    * * * *

    ascii_board:
    * * * *
       
    * *-*-*
       
    * * * *

    ascii_board:
    * * * *
        |
    * *-*-*
       
    * * * *

    ascii_board:
    * * *-*
        |
    * *-*-*
       
    * * * *

    ascii_board:
    * * *-*
        |
    * *-*-*
    |      
    * * * *

    ascii_board:
    * * *-*
        |
    * *-*-*
    |      
    *-* * *

    ascii_board:
    * * *-*
        |A|
    * *-*-*
    |      
    *-* * *

    ascii_board:
    * * *-*
    |   |A|
    * *-*-*
    |      
    *-* * *

    ascii_board:
    * * *-*
    |   |A|
    * *-*-*
    |     |
    *-* * *

    ascii_board:
    * * *-*
    |   |A|
    * *-*-*
    | |   |
    *-* * *

    ascii_board:
    * * *-*
    | | |A|
    * *-*-*
    | |   |
    *-* * *

    ascii_board:
    * * *-*
    | | |A|
    *-*-*-*
    |A|   |
    *-* * *

    ascii_board:
    * *-*-*
    | |A|A|
    *-*-*-*
    |A|   |
    *-* * *

    ascii_board:
    * *-*-*
    | |A|A|
    *-*-*-*
    |A| | |
    *-* * *

    ascii_board:
    * *-*-*
    | |A|A|
    *-*-*-*
    |A| |B|
    *-* *-*

    ascii_board:
    * *-*-*
    | |A|A|
    *-*-*-*
    |A|B|B|
    *-*-*-*

    ascii_board:
    *-*-*-*
    |B|A|A|
    *-*-*-*
    |A|B|B|
    *-*-*-*

    ```
