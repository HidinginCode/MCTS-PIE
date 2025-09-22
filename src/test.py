from map import Map

def map_creation_test():
    """Tests if map creation works as intended.
    """
    test_map = Map(map_dim=5)
    
    assert len(test_map.map) == 5, (f"Map dimension 1 has length {len(test_map.map)}, but should have 5!")
    for row in test_map.map:
        assert len(row) == 5, (f"Map dimension 2 row {row} has length {len(row)}, but should have 5")

    print(test_map.map)
    print("Map creation test passed.")

def main():
    """Main function that runs all the defined test methods.
    """
    map_creation_test()

if __name__ == "__main__":
    main()