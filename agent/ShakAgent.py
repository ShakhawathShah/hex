import socket
from random import choice
from time import sleep
import sys
from collections import deque


class NaiveAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.board = []
        self.colour = ""
        self.turn_count = 0
        self.fair_openers = [(0, 10), (1, 2), (1, 8), (2, 0), (2, 5), (2, 10), (3, 0), (3, 10), (4, 10), (5, 0), (5, 10), (6, 0), (7, 0), (7, 10), (8, 0), (8, 5), (8, 10), (9, 2), (9, 8), (10, 0)]
        self.fair_swaps = [(0, 2), (1, 1), (2, 0), (8, 10), (9, 9), (10, 8)]
        self.bad_swaps = [(0, 0), (0, 1), (1, 0), (9, 10), (10, 9), (10, 10)]
        self.depth = 2

    def run(self):
        """Reads data until it receives an END message or the socket closes."""

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if (self.interpret_data(data)):
                break

        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        # if the game ended, False otherwise.
        """Checks the type of message and responds accordingly. Returns True
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [
                    [0]*self.board_size for i in range(self.board_size)]

                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()

                    self.make_move()

        return False


    def make_move(self):
        """Uses the Minimax algorithm to make the best move."""
        best_score = -sys.maxsize
        best_move = None
        hardcoded_move = None
        possible_moves = self.get_possible_moves()

        if self.colour == "R":
            # Make a fair opening move, low chance of swapping
            if self.turn_count == 0:
                move = choice(self.fair_openers)
                self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))
                self.board[move[0]][move[1]] = self.colour
                self.turn_count += 1
                return
            if self.turn_count == 1 and (5, 5) in possible_moves:
                hardcoded_move = (5, 5)
            if self.turn_count == 2 and (3,6) in possible_moves:
                hardcoded_move = (3,6)
            if self.turn_count == 3 and (7,4) in possible_moves:
                hardcoded_move = (7,4)
            if self.turn_count == 4 and (8,2) in possible_moves:
                hardcoded_move = (8,2)

        if self.colour == "B":
            # Swap if move worthy of swapping
            if self.turn_count == 0:
                for move in self.bad_swaps:
                    if move not in possible_moves:
                        self.s.sendall(bytes(f"{5},{5}\n", "utf-8"))
                        self.board[5][5] = self.colour
                        self.turn_count += 1
                        return
                # Swap if opponent is making a fair opening move only 50% of the time
                for move in self.fair_swaps:
                    if move not in possible_moves and choice([0, 1]) == 1:
                        self.s.sendall(bytes("SWAP\n", "utf-8"))
                        self.turn_count += 1
                        return
                # Swap any other move not in fair or bad swaps
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.turn_count += 1
                return
            if self.turn_count == 1 and (5,5) in possible_moves:
                hardcoded_move = (5,5)
            if self.turn_count == 2 and (5,3) in possible_moves:
                hardcoded_move = (5,3)
            if self.turn_count == 3 and (5,7) in possible_moves:
                hardcoded_move = (5,7)
            if self.turn_count == 4 and (4,2) in possible_moves:
                hardcoded_move = (4,2)
            
        if hardcoded_move:
            print("Using hardcoded move")
            self.s.sendall(bytes(f"{hardcoded_move[0]},{hardcoded_move[1]}\n", "utf-8"))
            self.board[hardcoded_move[0]][hardcoded_move[1]] = self.colour
            self.turn_count += 1
            return

        # Try make a move using minimax
        _, best_move = self.minimax(self.board, self.depth, False, -sys.maxsize, sys.maxsize)

        if best_move == (None, None):
            # Make a random move if no best move is found
            print("No best move found, making random move!!!!")
            self.random_move()
        else:
            self.s.sendall(bytes(f"{best_move[0]},{best_move[1]}\n", "utf-8"))
            self.board[best_move[0]][best_move[1]] = self.colour
            self.turn_count += 1

    def get_possible_moves(self):
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] == 0]


    def minimax(self, board, depth, isMaximizingPlayer, alpha, beta):
        best_move = (None, None)
        if depth == 0:
            return self.evaluate(board), best_move

        if isMaximizingPlayer:
            maxEval = -sys.maxsize
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0:
                        board[i][j] = self.colour
                        score,_ = self.minimax(board, depth - 1, False, alpha, beta)
                        board[i][j] = 0
                        if(score > maxEval):
                            best_move = (i,j)
                        maxEval = max(maxEval, score)
                        alpha = max(alpha, score)
                        if beta <= alpha:
                            break
            return maxEval, best_move
        else:
            minEval = sys.maxsize
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0:
                        board[i][j] = self.opp_colour()
                        score,_ = self.minimax(board, depth - 1, True, alpha, beta)
                        board[i][j] = 0
                        if(score < minEval):
                            best_move = (i,j)
                        minEval = min(minEval, score)
                        beta = min(beta, score)
                        if beta <= alpha:
                            break
            return minEval, best_move

    def get_neighbors(self, position):
        x, y = position
        directions = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
        neighbors = [(x + dx, y + dy) for dx, dy in directions]
        return [(x, y) for x, y in neighbors if 0 <= x < self.board_size and 0 <= y < self.board_size]

            
    def numPlaysToWin(self, board, colour):
        queue = deque()
        distances = {(x, y): float('inf') for x in range(self.board_size) for y in range(self.board_size)}
        opponent = 'B' if colour == 'R' else 'R'

        # Add starting positions to the queue
        if colour == 'R':
            for i in range(self.board_size):
                if board[0][i] != opponent:
                    start = (0, i)            
                    distances[start] = 0 if board[i][0] == 'R' else 1
                    queue.append(start)
        else:
            for i in range(self.board_size):
                if board[i][0] != opponent:
                    start = (i, 0)
                    distances[start] = 0 if board[i][0] == 'B' else 1
                    queue.append(start)

        while queue:
            position = queue.popleft()
            for neighbor in self.get_neighbors(position):
                if board[neighbor[0]][neighbor[1]] != opponent:
                    new_dist = distances[position] + (1 if board[neighbor[0]][neighbor[1]] == 0 else 0)
                    if distances[neighbor] > new_dist:
                        distances[neighbor] = new_dist
                        queue.append(neighbor)
        if colour == 'R':
            return min(distances[(self.board_size - 1, i)] for i in range(self.board_size)) if any(
                distances[(self.board_size - 1, i)] != float('inf') for i in range(self.board_size)) else 1000
        else:
            return min(distances[(i, self.board_size - 1)] for i in range(self.board_size)) if any(
                distances[(i, self.board_size - 1)] != float('inf') for i in range(self.board_size)) else 1000
    

    def calculateCenterControl(self, board, color):
        center_x, center_y = len(board) // 2, len(board[0]) // 2
        center_control = 0

        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == color:
                    # Measure the Manhattan distance to the center
                    distance = abs(center_x - row) + abs(center_y - col)
                    center_control += distance

        return center_control
    

    def is_valid(self, x, y, size):
        return 0 <= x < size and 0 <= y < size

    def dfs(self, x, y, grid, visited, player, chain_id, chain_sizes):
        size = len(grid)
        directions = [(0, 1), (1, 0), (-1, 1), (1, -1), (-1, 0), (0, -1)]

        visited[x][y] = chain_id
        chain_sizes[chain_id] += 1

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny, size) and not visited[nx][ny] and grid[nx][ny] == player:
                self.dfs(nx, ny, grid, visited, player, chain_id, chain_sizes)

    def count_chains_and_sizes_for_player(self, grid, player):
        size = len(grid)
        visited = [[0 for _ in range(size)] for _ in range(size)]
        chain_count = 0
        chain_sizes = {}

        for i in range(size):
            for j in range(size):
                if grid[i][j] == player and not visited[i][j]:
                    chain_count += 1
                    chain_sizes[chain_count] = 0
                    self.dfs(i, j, grid, visited, player, chain_count, chain_sizes)

        return chain_count, chain_sizes


    def min_max_scaling(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    
    def evaluate(self, board):
        opponent_plays_to_win = self.numPlaysToWin(board, self.opp_colour())
        agent_plays_to_win = self.numPlaysToWin(board, self.colour)

        opponent_center_control = self.calculateCenterControl(board, self.opp_colour())
        agent_center_control = self.calculateCenterControl(board, self.colour)

        maximise_difference_in_connections_per_chain = 0
        # if (self.turn_count > math.ceil(self.board_size/2)):

        opp_chains_count, opp_chain_sizes = self.count_chains_and_sizes_for_player(board, self.opp_colour())
        agent_chains_count, agent_chain_sizes = self.count_chains_and_sizes_for_player(board, self.colour)

        # minimise (dont want to connect their bridges)
        opp_connections_per_chain = sum(opp_chain_sizes.values())/opp_chains_count
        # maximise (want to connect our bridges)
        agent_connections_per_chain = sum(agent_chain_sizes.values())/agent_chains_count
        # print(f"Connections per chain\nOpponent: {opp_connections_per_chain}\nAgent: {agent_connections_per_chain}")

        # final scoring for jayan logic - largest = x/1 upper lower =  
        maximise_difference_in_connections_per_chain = (agent_connections_per_chain)/opp_connections_per_chain
        # ADJUSTMENT - if our chain is longer than theirs, be more offensive, else be defensive(block)
        # maximise_difference_in_connections_per_chain *= agent_chains_count/opp_chains_count
        # print(maximise_difference_in_connections_per_chain)

        scaled_plays_to_win = self.min_max_scaling((opponent_plays_to_win - agent_plays_to_win), 0, 11)
        scaled_centre_control = self.min_max_scaling((opponent_center_control - agent_center_control), 15, 180)

        plays_to_win_weight = 1.0
        centre_control_weight = 1.0
        difference_in_connections_per_chain_weight = 0.5

        # Reduce weight of centre control after turn 5
        if self.turn_count >= 5:
            centre_control_weight = 0
        # Reduce weight of difference in connections per chain if agent is close to winning
        if agent_plays_to_win <= 3:
            difference_in_connections_per_chain_weight = 0
        # Increase weight of difference in connections per chain if opponent is close to winning
        if opponent_plays_to_win <= 3:
            difference_in_connections_per_chain_weight = 1.0


        total_score = (plays_to_win_weight * scaled_plays_to_win) + (centre_control_weight * scaled_centre_control) + (difference_in_connections_per_chain_weight * maximise_difference_in_connections_per_chain)

        return total_score
    
    def is_winner(self, board):
        def dfs(x, y, target, visited):
            """Depth-first search to check if a player has connected their sides."""
            if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]) or board[x][y] != target or (x, y) in visited:
                return False

            visited.add((x, y))

            # Check if the player has connected their sides
            if (target == 'R' and y == len(board[0]) - 1) or (target == 'B' and x == len(board) - 1):
                return True

            # Explore neighbors
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, -1)]
            return any(dfs(x + dx, y + dy, target, visited) for dx, dy in directions)

        # Check if 'Red' has won
        if any(dfs(0, idx, 'R', set()) for idx in range(len(board[0]))):
            return 'R'

        # Check if 'Blue' has won
        if any(dfs(idx, 0, 'B', set()) for idx in range(len(board))):
            return 'B'

        return None  # No winner
    
    def random_move(self):
        """Makes a random move from the available pool of choices."""
        print("Making random move")
        choices = self.get_possible_moves()
        if choices:
            pos = choice(choices)
            self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
            self.board[pos[0]][pos[1]] = self.colour
            self.turn_count += 1


    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
        current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"


if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()
