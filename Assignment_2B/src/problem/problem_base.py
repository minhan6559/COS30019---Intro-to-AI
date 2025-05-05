class ProblemBase:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goals=None):
        """The constructor specifies the initial state, and possibly goals
        state(s). Your subclass's constructor can add other arguments."""
        self.initial = initial
        # Always store goals as a list for consistency
        if goals is None:
            self.goals = []
        elif isinstance(goals, list):
            self.goals = goals
        else:
            self.goals = [goals]

    def actions(self, state):
        raise NotImplementedError

    def goal_test(self, state):
        raise NotImplementedError

    def path_cost(self, c, state1, state2):
        raise NotImplementedError
