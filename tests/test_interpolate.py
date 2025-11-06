# Test interpolation routines


def test_find_nearest_points_pair():
    import numpy as np

    from coordarray.interpolation import find_nearest_points_pair

    array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([2.5, 4.5])

    indices = find_nearest_points_pair(array, values)

    assert indices.shape == (2, 2)
    assert np.array_equal(indices[0], [1, 2])  # Nearest points for 2.5
    assert np.array_equal(indices[1], [3, 4])  # Nearest points for 4.5


def test_find_nearest_points_pair_single_value():
    import numpy as np

    from coordarray.interpolation import find_nearest_points_pair

    array = np.array([10, 20, 30, 40, 50])
    value = 33

    indices = find_nearest_points_pair(array, value)

    assert indices.shape == (1, 2)
    assert np.array_equal(indices[0], [2, 3])  # Nearest points for 33


def test_find_nearest_points_pair_edge_cases():
    import numpy as np

    from coordarray.interpolation import find_nearest_points_pair

    array = np.array([0, 100, 200, 300, 400])
    values = np.array([-50, 0, 450])

    indices = find_nearest_points_pair(array, values)

    assert indices.shape == (3, 2)
    assert np.array_equal(indices[0], [0, 0])  # Nearest points for -50
    assert np.array_equal(indices[1], [0, 1])  # Nearest points for 0
    assert np.array_equal(indices[2], [4, 4])  # Nearest points for 450


def test_coordarray_subset():
    import numpy as np

    from coordarray import coordarray
    from coordarray.interpolation import coordarray_subset

    data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    coords = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([10.0, 20.0, 30.0])}
    arr = coordarray(data, **coords)

    new_coords = {"x": np.array([1.5]), "y": np.array([15.0])}

    subset = coordarray_subset(arr, new_coords)

    assert subset.shape == (2, 2)
