import heapq
import math
from collections import Counter, deque
from functools import reduce, cache
import random


@cache
def get_neighbors(row, col): # generates invalid tiles but we don't care
    directions = [
        (0, -1),   # Izquierda
        (0, 1),    # Derecha
        (-1, 0),   # Arriba
        (1, 0),    # Abajo
        (-1, 1),   # Arriba derecha
        (1, -1)    # Abajo izquierda
    ]
    return [(row + dr, col + dc) for dr, dc in directions]


class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal` states
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        return state == self.goal

    def action_cost(self, s, a, s1):
        return 1

    def h(self, node):
        return 0

    def __str__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    "A Node in a search tree."

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(
            state=state, parent=parent, action=action, path_cost=path_cost
        )

    def __repr__(self):
        return "<{}>".format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node(
    "failure", path_cost=math.inf
)  # Indicates an algorithm couldn't find a solution.
cutoff = Node(
    "cutoff", path_cost=math.inf
)  # Indicates iterative deepening search was cut off.


def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)


def best_first_search(problem, f):
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure


def g(n):
    return n.path_cost


def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


def weighted_astar_search(problem, h=None, weight=1.4):
    """Search nodes with minimum f(n) = g(n) + weight * h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + weight * h(n))


# implementation
def opponent(player_id):
    return 3 - player_id


class HexProblem(Problem):
    """Formal problem definition for Hex game using A* search"""

    def __init__(self, board, player_id, **kwargs):
        self.board = board
        self._default = board.clone().board
        self.player_id = player_id
        self.opponent_id = opponent(player_id)
        super().__init__(initial=board, **kwargs)

    def actions(self, state):
        """Return all empty cells as possible actions"""
        return state.get_possible_moves()

    def action_cost(self, s, a, s1):
        return 1

    def result(self, state, action):
        """Return new board after making a move"""
        new_board = state.clone()
        row, col = action
        new_board.place_piece(row, col, self.player_id)

        return new_board

    def h(self, node):
        """
        Try to block the opponent. That is, prioritize paths closer to the opponent.
        h((x, y)) = (euclidian_dist_to_goal - manhattan_to_opponent)
        """
        board = node.state
        total = manhattan(node.state, self.player_id)
        return total

        # xxx
        ai_distance = 0
        opponent_distance = self.shortest_path_distance(board, self.opponent_id)
        return opponent_distance - ai_distance

    def is_goal(self, state):
        return state.check_connection(self.player_id)

    def __repr__(self):
        return self.__str__()


# h
def nun(board, player_id):
    return 0


# kuso
def manhattan(board, player_id):
    pos = sorted(board.player_positions[player_id], reverse=player_id != 2)
    if not pos:
        return board.size
    if player_id == 2:
        start, end = (0, pos[0][1]), (board.size - 1, pos[-1][1])
    else:  # player_id == 1
        start, end = (pos[-1][0], board.size - 1), (pos[0][0], 0)
    if end not in pos:
        pos.append(end)

    total = 0
    v = start
    for u in pos:
        total += abs(v[0] - u[0]) + abs(v[1] - u[1])
        v = u

    return -total

@cache
def is_valid(coords, size):
    return (
        0 <= coords[0]
        and coords[0] < size
        and 0 <= coords[1]
        and coords[1] < size
    )

def exhaust_coords(own_pieces, coords, size, seen):
    r, c = coords
    total = []
    seen.add(coords)
    for n in get_neighbors(r, c):
        if is_valid(n, size) and n in own_pieces and n not in seen:
            seen.add(n)
            total += exhaust_coords(own_pieces, n, size, seen)
    return [coords] + total

def _distance_heuristic(board, player_id):
    """
    We have an advantage if:
    - We have to place few pieces to complete a path.
        That is, we don't have to place many pieces to
    - The distance between our connected sets of pieces, in average, is low
       The intuition is that, if our pieces are close to each other, we can create
       more bridges and block the opponent
    """
    # Now, how to implement this
    # Consider our board is a graph, G
    # - Any edge from/to an opponent's node is uncrossable: w = +inf
    # - Any edge from/to our nodes can be crossed without placing a(nother) piece: w = 0
    # - w = 1 in other case
    #
    # Let's consider the modified graph G_p, with two extra "phantom" nodes, (u and v).
    # These are connected to each one of the borders (up, down for player 2; left,right for player 1).
    # By definition, any validーemphasis in validーpath is a winning path in Hex for our player.
    # This path exists because, if any of the phantom nodes is fully blockedーthat is, all its
    # neighbors are owned by the other playerーthis would create a winning path for our opponent,
    # meaning the game is already over.
    # Now let's remove all opponent's nondes with their edges. This will never disconnect any of the
    # valid paths because their cost is +inf.
    # And let's also "compress" all our nodes as well. By compressing them (that is, visiting their nodes
    # without increasing the cost) we don't affect the cost of the shortest path.
    # This creates a digraph where all the costs are exactly 1. Now the shortest path can be found using BFS.
    # The cost of the shortest path is exactly how many pieces we need to place to complete the path
    # and create a zero-cost winning path by our definitions. And minimum number of pieces required to
    # create a winning path is the cost of the shortest path.
    # To prove this implication, consider we can create a winning path using less pieces. The winning path
    # is connected to both phantom nodes by definition. Thus, we just created a path with less cost than
    # the shortest path. This is a contradiction, thus, the array created by BFS yields
    # the shortest winning path and the cost of this path is the least amount of pieces needed to
    # complete the path.
    #
    # Now, to compute the average distance between each one of our nodes, we only need to iterate the shortest path
    # Here is the pseudocode for the full algorithm.
    size = board.size
    opponent_pieces = board.player_positions[opponent(player_id)]
    own_pieces = board.player_positions[player_id]

    opponent_id = opponent(player_id)

    # Phantom nodes for BFS
    if player_id == 1:  # left-right
        phantom_nodes = [(-1, -1), (-1, size)]
    else:  # top-botton
        phantom_nodes = [(-1, -1), (size, -1)]

    start_phantom, end_phantom = phantom_nodes

    visited = set()
    seen = set()
    queue = deque()
    parent = {}
    distance = {}
    terminal = set()

    # we do it here because the phantom nodes do not follow the adj rules
    visited.add(start_phantom)
    distance[start_phantom] = 0

    distance[end_phantom] = float("inf")
    parent[end_phantom] = None

    if player_id == 1:  # connect to left edges
        for r in range(size):
            left, right = (r, 0), (r, size - 1)
            if left not in opponent_pieces and left not in own_pieces:
                queue.append(left)
                seen.add(left)
                parent[left] = start_phantom
                distance[left] = 0 if left in own_pieces else 1
            elif left in own_pieces: # compress
                all_coords = exhaust_coords(own_pieces, left, size, seen)
                for crd in all_coords:
                    parent[crd] = start_phantom
                    distance[crd] = 0
                    queue.appendleft(crd)

            # connect other phantom
            if right not in opponent_pieces:
                terminal.add(right)
                #distance[right] = 0 if right in own_pieces else 1

    else:  # connect to top edge
        for c in range(size):
            top, bottom = (0, c), (size - 1, c)
            if top not in opponent_pieces and top not in own_pieces:
                queue.append(top)
                seen.add(top)

                parent[top] = start_phantom
                distance[top] = 0 if top in own_pieces else 1
            elif top in own_pieces:
                all_coords = exhaust_coords(own_pieces, top, size, seen)
                for crd in all_coords:
                    parent[crd] = start_phantom
                    distance[crd] = 0
                    queue.appendleft(crd)

            if bottom not in opponent_pieces:
                terminal.add(bottom)
                #distance[bottom] = 0 if bottom in own_pieces else 1

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if (r, c) in terminal and end_phantom not in seen:
            distance[end_phantom] = distance[(r, c)]
            parent[end_phantom] = (r, c)
            seen.add(end_phantom)
            #break

        for coords in get_neighbors(r, c):
            if coords in seen:
                continue

            if coords in own_pieces:
                # no cost
                all_coords = exhaust_coords(own_pieces, coords, size, seen)
                for crd in all_coords:
                    parent[crd] = (r, c)
                    distance[crd] = distance[(r, c)] # zero cost
                    # appendleft because these are at distance[(r, c)], we can
                    # not put them next to the nodes at distance + 1
                    queue.appendleft(crd)
            elif coords in opponent_pieces:
                # can not pass
                continue
            elif is_valid(coords, size):
                parent[coords] = (r, c)
                distance[coords] = distance[(r, c)] + 1
                queue.append(coords)
                seen.add(coords)

    return parent, distance, end_phantom

def find_average_distance(parent, end_phantom, own_pieces):
    """
    Find average distance between connected components formed by
    our pieces in the shortest path... in other words, average of the
    white spaces in the shortest path
    """
    # idea: if the current node is ours, reset counter and add the current
    # distance to an array
    # find the average of the array once finished
    node = end_phantom
    counter = 0
    distances = []
    while parent.get(node):
        if counter and parent[node] in own_pieces:
            distances.append(counter)
            counter = 0
        elif parent[node] not in own_pieces:
            counter += 1
        node = parent[node]
    if not distances:
        return counter
    return sum(distances)/len(distances)

def near_count(parent, end_phantom, own_pieces):
    """
    Gain advantage when having pieces close to each other
    """
    node = end_phantom
    counter = 0
    distances = []
    while parent.get(node):
        if counter and parent[node] in own_pieces:
            distances.append(counter)
            counter = 0
        elif parent[node] not in own_pieces:
            counter += 1
        node = parent[node]
    if not distances:
        return -counter
    c = Counter(distances)
    return 0.5*c[1] + 0.2*c[2]

def full_near_count(parent, end_phantom, own_pieces, opponent_pieces):
    """
    Gain advantage when:
    - having pieces close to each other
    - being near to the center of the board
    - being next to an opponent's tile
    """
    node = end_phantom
    counter = 0
    distances = []

    near_opponent = 0
    while parent.get(node):
        if counter and parent[node] in own_pieces:
            distances.append(counter)
            counter = 0
            x, y = parent[node]
            for neigh in get_neighbors(x, y):
                near_opponent += 1
        elif parent[node] not in own_pieces:
            counter += 1
        node = parent[node]
    if not distances:
        return -counter
    c = Counter(distances)
    return 0.5*c[1] + 0.2*c[2] + 0.01*near_opponent



# distance h function
def distance_heuristic(board, player_id):
    parent, distance, end_node = _distance_heuristic(board, player_id)

    #return -distance[end_node]
    return 2*(board.size - distance[end_node]) # placed pieces to make the path shorter

def average_distance_heuristic(board, player_id):
    parent, distance, end_node = _distance_heuristic(board, player_id)
    own_pieces = board.player_positions[player_id]

    #return -find_average_distance(parent, end_node, own_pieces)
    return near_count(parent, end_node, own_pieces)

def full_distance_heuristic(board, player_id):
    parent, distance, end_node = _distance_heuristic(board, player_id)
    own_pieces = board.player_positions[player_id]
    opponent_pieces = board.player_positions[opponent(player_id)]

    #return -distance[end_node]*0.5 + -find_average_distance(parent, end_node, own_pieces)
    return (
        2*(board.size - distance[end_node])
        #+ near_count(parent, end_node, own_pieces)
        + full_near_count(parent, end_node, own_pieces, opponent_pieces)
    )


def adversarial_heuristic(advantages: 'List[Callable]'):
    """
    Idea: an advantage for me it's a disadventage for the other player and
     vice-versa.
    """
    def _adv(board, player_id):
        total = 0
        for adv in advantages:
            total += adv(board, player_id)
            total -= adv(board, opponent(player_id))

        return total

    return _adv


def minimax(board, depth, alpha, beta, maximising, player_id, heuristic):
    # base
    if (
        depth == 0
        or board.check_connection(player_id)
        or board.check_connection(opponent(player_id))
    ):
        return heuristic(board, player_id), None

    other = opponent(player_id)

    valid_moves = board.get_possible_moves()
    best_move = None

    bound = float("-inf") if maximising else float("inf")
    for move in valid_moves:
        new_board = board.clone()
        x, y = move
        new_board.place_piece(x, y, player_id if maximising else opponent(player_id))

        val, _ = minimax(
            new_board,
            depth - 1,
            alpha,
            beta,
            not maximising,
            player_id,
            heuristic,
        )

        # we update bounds to find the min-max and prune
        if maximising and val > bound:
            bound = val
            best_move = move
            alpha = max(alpha, bound)
        elif not maximising and val < bound:
            bound = val
            best_move = move
            beta = min(beta, bound)

        # prune
        if maximising and bound >= beta:
            break
        elif not maximising and bound <= alpha:
            break
    return bound, best_move
