from first_test import factor, arrival, bonus, reward, duration, time_bonus
from trivial_sample_solution import assign_tasks

def sample_assign_tasks(factor, arrival, bonus, reward, duration, time_bonus):
    """Assign tasks to processors."""
    """Simple dumb strategy, guaranteed to produce valid solutions."""
    """All tasks are assigned to processor 0, each are scheduled"""
    """max(duration) ticks apart."""
    num_tasks = len(factor)
    max_arrival = max(arrival)
    max_duration = max(duration)
    schedule = []
    for i in range(num_tasks):
        schedule.append((0, max_arrival + i * max_duration))
    return schedule

def assign_tasks(factor, arrival, bonus, reward, duration, time_bonus):
    import numpy as np
    import random

    num_tasks = len(factor)
    max_arrival = max(arrival)
    max_duration = max(duration)
    schedule = []

    def task_score_func(p, t):

        if t < arrival[p] + time_bonus[p]:
            return factor[p][p] * (bonus[p] + reward[p] * duration[p] / (duration[p] + t - arrival[p]))
        else:
            return factor[p][p] * reward[p] * duration[p] / (duration[p] + t - arrival[p])

#    def compatibility():
#        compatibility_matrix = np.zeros((num_tasks, num_tasks))
#        for i in range(num_tasks):
#            for j in range(num_tasks):
#                if i != j:
#                    compatibility_matrix[i][j] = arrival[j] - arrival[i] - duration[i]
#                else:
#                    compatibility_matrix[i][j] = np.inf
#                if compatibility_matrix[i][j] < 0:
#                    compatibility_matrix[i][j] = np.inf
#        print(compatibility_matrix)
#        min_row_index = np.argmin(compatibility_matrix, axis=0)
#        min_row_values = np.min(compatibility_matrix, axis=0)
#
#        sorted_indices_flat = np.argsort(compatibility_matrix, axis=None)
#
#        sorted_indices_3d = np.unravel_index(sorted_indices_flat, compatibility_matrix.shape)
#
#        sorted_indices_list = list(zip(*sorted_indices_3d))
#
#        print(sorted_indices_list)
#        sorted_values = compatibility_matrix[sorted_indices_3d]
#        print(sorted_values)
#
#
#        current_min = 0
#        sorted_tasks = []

        return compatibility_matrix


    class TreeNode:
        def __init__(self, schedule, parent=None):
            self.parent = parent
            self.state = [(None, None) for _ in range(num_tasks)]
            self.children = []
            self.value = 0
            self.visits = 0

        def best_child(self, exploration_weight=1.4):
                choices_weights = [
                    (child.value / child.visits) + exploration_weight * math.sqrt(
                        2 * math.log(self.visits) / child.visits
                    )
                    for child in self.children
                ]
                return self.children[np.argmax(choices_weights)]

        def is_fully_expanded(self, ):
            num_tasks_scheduled = len([x for x in self.state if x[0] is not None])
            return num_tasks == num_tasks_scheduled

        def expand(self):
            actions = get_possible_actions(self.state)
            for action in actions:
                new_state = take_action(action, self.state)
                child_node = TreeNode(new_state, parent=self)
                self.children.append(child_node)

        def update(self, score):
            self.visits += 1
            self.value += score

    def is_terminal(state):
        for a in state:
            if a[0] is None:
                return False
        return True

    def take_action(action, state):
        state[action[0]] = (0, action[1])
        return state

    def get_possible_actions(state):
        if len([x for x in state if x[0] is not None]) == 0:
            current_time = min(arrival)
        else:
            tasks = [i for i, x in enumerate(state) if x[0] is not None]
            times = [x[1] for x in state if x[0] is not None]
            max_time_task = tasks[np.argmax(times)]
            current_time = max(times) + duration[max_time_task]
        possible_actions = []
        existing_tasks = [i for i, x in enumerate(state) if x[0] is not None]
        while len(possible_actions) == 0:
            for i in range(num_tasks):
                if arrival[i] <= current_time:
                    if i not in existing_tasks:
                        possible_actions.append((i, current_time))
            current_time += 1
        return possible_actions

    class MCTS:
        def __init__(self, root):
            self.root = root

        def select(self, node):
            while not is_terminal(node.state) and node.is_fully_expanded():
                node = node.best_child()
            return node

        def expand(self, node):
            if not is_terminal(node.state) and not node.is_fully_expanded():
                node.expand()
            return random.choice(node.children)

        def simulate(self, node):
            current_state = node.state
            while not is_terminal(current_state):
                actions = get_possible_actions(current_state)
                action = random.choice(actions)
                current_state = take_action(action, current_state)
            print(is_valid_schedule(current_state))
            simulated_score = self.schedule_score_func(current_state)
#            print(current_state)
#            print(simulated_score)
            return simulated_score

        def backpropagate(self, node, score):
            while node is not None:
                node.update(score)
                node = node.parent

        def run(self, num_simulations):
            for _ in range(num_simulations):
                node = self.select(self.root)
                if not is_terminal(node.state):
                    node = self.expand(node)
                score = self.simulate(node)
                self.backpropagate(node, score)

        def best_action(self):
            return max(self.root.children, key=lambda child: child.visits)

        def schedule_score_func(self, schedule):
            score = 0

            for i in range(len(schedule)):
                p, t = schedule[i]

                if t < arrival[i] +time_bonus[i]:
                    score += factor[i][p] * (bonus[i] + reward[i] * duration[i] / (duration[i] + t - arrival[i]))
                else:
                    score += factor[i][p] * reward[i] * duration[i] / (duration[i] + t - arrival[i])

            return score

    def is_valid_schedule(schedule):
    # sort a list of tuples by the second element in each tuple and return the indexes
        task_indices = [i[0] for i in sorted(enumerate(schedule), key=lambda x:x[1][1])]
        print(task_indices)

        for i in range(num_tasks):
            for j in range(i+1, num_tasks):
                if schedule[task_indices[i]][1] + duration[task_indices[i]] > schedule[task_indices[j]][1]:
                    print("Invalid schedule")
                    return False
        return True


    root = TreeNode(schedule)
    mcts = MCTS(root)
    mcts.run(1000)
    best_action = mcts.best_action()



    return schedule



"""Resulting schedule."""
#resulting_schedule = sample_assign_tasks(factor, arrival, bonus, reward, duration, time_bonus)
#"""Score of the resulting schedule."""
#score = score_func(resulting_schedule, factor, arrival, bonus, reward, duration, time_bonus)
#print(score)

assign_tasks(factor, arrival, bonus, reward, duration, time_bonus)

'''base score for test1: 184.28602327280973'''

