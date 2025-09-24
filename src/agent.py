"""This module holds the agent class, which collects metrics for the experiments."""

from copy import deepcopy

class Agent():
    """This class represents the agent and collects metrics for the experiments.
    """

    # Metric Variables
    step_count = 0
    amount_of_shifts = 0
    energy_consumption = 0.0
    identificator = 111111111
    path = []

    def __init__(self):
        """Init function for the Agent class."""
        self.identificator = id(self)
        self.energy_consumption = 0.0

    def __str__(self) -> str:
        """String function for the agent.

        Returns:
            str: String containing all metrics and identifier
        """
        return (
            f"Stepcount: {self.step_count}\n"
            f"Amount of shifts: {self.amount_of_shifts}\n"
            f"Energy Consumption: {self.energy_consumption}\n"
            f"Identificator: {self.identificator}"
        )

    def get_step_count(self) -> int:
        """Returns an agents step count.

        Returns:
            int: Step count of the agent.
        """
        return self.step_count

    def get_amount_of_shifts(self) -> int:
        """Returns amount of shifts and agent performed.

        Returns:
            int: Amount of shifts performed by the agent.
        """
        return self.amount_of_shifts

    def get_energy_consumption(self) -> float:
        """Returns amount of energy used by the agent.

        Returns:
            float: Amount of energy used by the agent.
        """
        return self.energy_consumption

    def get_idenificator(self) -> int:
        """Returns identificator of the agent.

        Returns:
            int: Agent identificator
        """
        return self.identificator

    def get_path(self) -> list:
        """Returns path of the agent.

        Returns:
            list: Agent path.
        """
        return self.path

    def set_path(self, path: list) -> None:
        """Sets an agents path.

        Args:
            path (list): New path for agent
        """
        self.path = deepcopy(path)
