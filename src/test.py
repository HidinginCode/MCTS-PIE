""" Test Module.

This module contains all the tests needed for this project.
"""

from map import Map
from agent import Agent

def map_creation_test():
    """Tests if map creation works as intended.
    """
    test_map = Map(map_dim=5)

    assert len(test_map.map) == 5, (
        f"Map dimension 1 has length {len(test_map.map)},but should have 5!"
    )

    for row in test_map.map:
        assert len(row) == 5, (
            f"Map dimension 2 row {row} has length {len(row)}, but should have 5"
            )

    print("Map creation test passed.")

def agent_creation_test():
    """This function tests the creation of an Agent
    """
    new_agent = Agent()
    assert isinstance(new_agent.get_idenificator(), int), (
        "Agent get_identificator method did not return int."
    )

    assert isinstance(new_agent.get_amount_of_shifts(), int), (
        "Agents get_amount_of_shifts method did not return int."
    )

    assert isinstance(new_agent.get_energy_consumption(), float), (
        "Agents get_energy_consumption method did not return float."
    )

    assert isinstance(new_agent.get_path(), list), (
        "Agents get_path method did not return list."
    )

    new_agent.set_path([[0,0], [1,1]])

    assert new_agent.get_path() == [[0,0], [1,1]], (
        f"""After setting new path, agents get_path method should have returned
        {[[0,0], [1,1]]}, but returned {new_agent.get_path()}."""
    )

    print("Agent creation test passed.")

def main():
    """Main function that runs all the defined test methods.
    """
    map_creation_test()
    agent_creation_test()

if __name__ == "__main__":
    main()
