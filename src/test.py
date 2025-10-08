""" Test Module.

This module contains all the tests needed for this project.
"""

from map import Map
from agent import Agent
from controller import Controller

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

    assert isinstance(test_map.get_identificator(), int), (
        "Maps get_identificator method did not return int."
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

def controller_creation_test():
    """Method that tests the creation of the controller.
    """
    test_map = Map(map_dim=5)
    test_agent = Agent()
    test_controller = Controller(map_copy=test_map, current_agent=test_agent, start_pos=(0,0))

    assert isinstance(test_controller.get_map_copy(), Map), (
        "Controllers get_map_copy methdo did not return Map object."
    )

    assert isinstance(test_controller.get_current_agent(), Agent), (
        "Controllers get_current_agent method did not return Agent object."
    )

    assert isinstance(test_controller.get_identificator(), int), (
        "Controllers get_identificator method did not return int."
    )

    test_map2 = Map(map_dim=10)
    test_controller.set_map_copy(map_copy=test_map2)

    old_map_id = test_map.get_identificator()
    new_map_id = test_controller.get_map_copy().get_identificator()

    assert new_map_id != old_map_id, (
        "Newly set map should have a different identificator than the old map."
    )


    test_agent2 = Agent()
    test_controller.set_current_agent(current_agent=test_agent2)

    old_agent_id = test_agent.get_idenificator()
    new_agent_id = test_controller.get_current_agent().get_idenificator()

    assert old_agent_id != new_agent_id, (
        "Newly set agent should have a different identificator than the old agent."
    )

    assert test_controller.get_current_agent_position() == (0,0), (
        "Start position for the agent was set wrong on creation of the controller."
    )

    test_controller.set_current_agent_position((1,1))

    assert test_controller.get_current_agent_position() == (1,1), (
        "Controllers set_current_agent_position method did not set position correctly."
    )

    assert test_controller.is_valid_position((0,0)), (
        "Controllers is_valid_position method did not identify (0,0) as a valid position."
    )

    assert not test_controller.is_valid_position((-1,-1)), (
        "Controllers is_valid_position method did not identify (-1,-1) as an invalid position."
    )

    print("Controller creation test passed.")

def main():
    """Main function that runs all the defined test methods.
    """
    map_creation_test()
    agent_creation_test()
    controller_creation_test()

if __name__ == "__main__":
    main()
