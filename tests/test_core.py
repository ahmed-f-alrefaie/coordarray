"""Test coordinate array class."""

import numpy as np
import pytest
from astropy import units as u

from coordarray.core import coordarray


def test_coordarray_init_nparray():
    """Test initialization of coord array."""
    data = np.arange(10)
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)
    assert coord_array.shape == (10,)
    assert coord_array.ndim == 1
    assert coord_array.size == 10
    assert coord_array.data.shape == (10,)
    assert coord_array.data.ndim == 1
    assert coord_array.data.size == 10
    assert coord_array.coords["x"].shape == (10,)
    assert coord_array.coords["x"].ndim == 1
    assert coord_array.coords["x"].size == 10
    assert coord_array.dtype == np.dtype("int64")


def test_coordarray_init_quantity():
    """Test initialization of coord array."""
    data = np.arange(10) << u.K
    coords = {"x": np.arange(10) << u.m}
    coord_array = coordarray(data, coords)
    assert coord_array.shape == (10,)
    assert coord_array.ndim == 1
    assert coord_array.size == 10
    assert coord_array.data.shape == (10,)
    assert coord_array.data.ndim == 1
    assert coord_array.data.size == 10
    assert coord_array.coords["x"].shape == (10,)
    assert coord_array.coords["x"].ndim == 1
    assert coord_array.coords["x"].size == 10
    assert coord_array.dtype == np.float64
    assert coord_array.value.unit == u.K


def test_coordarray_op_nparray():
    """Test operations for coord array."""
    data = np.arange(10)
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)

    assert np.all((data == coord_array).data)

    np.testing.assert_array_equal(coord_array + 1, data + 1)
    np.testing.assert_array_equal(coord_array + 1, data + 1)
    np.testing.assert_array_equal(coord_array - 1, data - 1)
    np.testing.assert_array_equal(coord_array * 2, data * 2)
    np.testing.assert_array_equal(coord_array / 2, data / 2)
    np.testing.assert_array_equal(coord_array**2, data**2)
    np.testing.assert_array_equal(coord_array % 2, data % 2)
    np.testing.assert_array_equal(coord_array // 2, data // 2)
    np.testing.assert_array_equal(coord_array << 2, data << 2)
    np.testing.assert_array_equal(coord_array >> 2, data >> 2)
    np.testing.assert_array_equal(coord_array & 2, data & 2)
    np.testing.assert_array_equal(coord_array | 2, data | 2)
    np.testing.assert_array_equal(coord_array ^ 2, data ^ 2)
    np.testing.assert_array_equal(coord_array**2, data**2)


def test_coordarray_op_coordarray():
    """Test operations for coord array."""
    data = np.arange(10) + 1
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)
    coord_array2 = coordarray(data, coords)

    np.testing.assert_array_equal((coord_array + coord_array2).data, data * 2)
    np.testing.assert_array_equal((coord_array - coord_array2).data, np.zeros(10))
    np.testing.assert_array_equal((coord_array * coord_array2).data, data**2)
    np.testing.assert_array_equal((coord_array / coord_array2).data, np.ones(10))
    np.testing.assert_array_equal((coord_array**coord_array2).data, data**data)
    np.testing.assert_array_equal((coord_array % coord_array2).data, np.zeros(10))
    np.testing.assert_array_equal((coord_array // coord_array2).data, np.ones(10))
    np.testing.assert_array_equal((coord_array << coord_array2).data, data << data)
    np.testing.assert_array_equal((coord_array >> coord_array2).data, data >> data)
    np.testing.assert_array_equal((coord_array & coord_array2).data, data & data)
    np.testing.assert_array_equal((coord_array | coord_array2).data, data | data)


def test_coordarray_op_coordarray_mismatch():
    """Test operations for coord array."""
    data1 = np.arange(1, 201).reshape(10, 20)
    data2 = np.arange(1, 201).reshape(20, 10)

    coords1 = {"x": np.arange(10), "y": np.arange(20)}
    coords2 = {"y": np.arange(20), "x": np.arange(10)}
    coord_array = coordarray(data1, coords1)
    coord_array2 = coordarray(data2, coords2)

    res = coord_array + coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 + data2.T)

    res = coord_array - coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 - data2.T)

    res = coord_array * coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 * data2.T)

    res = coord_array / coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 / data2.T)

    res = coord_array**coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1**data2.T)

    res = coord_array % coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 % data2.T)

    res = coord_array // coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 // data2.T)

    res = coord_array << coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 << data2.T)

    res = coord_array >> coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 >> data2.T)

    res = coord_array & coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 & data2.T)

    res = coord_array | coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data1 | data2.T)


def test_operator_coordarray_new_coords():
    """Test if new coords are included in operations."""
    data = np.arange(10)
    data2 = np.arange(20) + 100
    coord_array1 = coordarray(data, {"x": np.arange(10)})
    coord_array2 = coordarray(data2, {"y": np.arange(20)})

    res = coord_array1 + coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data[:, None] + data2[None, :])

    res = coord_array1 - coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data[:, None] - data2[None, :])

    res = coord_array1 * coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_array_equal(res.data, data[:, None] * data2[None, :])

    res = coord_array1 / coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] / data2[None, :])

    res = coord_array1**coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] ** data2[None, :])

    res = coord_array1 % coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] % data2[None, :])

    res = coord_array1 // coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] // data2[None, :])

    res = coord_array1 << coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] << data2[None, :])

    res = coord_array1 >> coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] >> data2[None, :])

    res = coord_array1 & coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] & data2[None, :])

    res = coord_array1 | coord_array2
    assert res.shape == (10, 20)
    assert res.axes == ("x", "y")
    np.testing.assert_allclose(res.data, data[:, None] | data2[None, :])


def test_stress_multi_axis():
    """Stress multi axis operations."""
    data = np.random.rand(3, 4, 5, 3, 2, 3, 4, 5)
    coords = {
        "x": np.arange(3),
        "y": np.arange(4),
        "z": np.arange(5),
        "a": np.arange(3),
        "b": np.arange(2),
        "c": np.arange(3),
        "d": np.arange(4),
        "e": np.arange(5),
    }

    data2 = np.random.rand(5, 4, 3, 3, 5, 2, 4, 2)
    coords2 = {
        "e": np.arange(5),
        "d": np.arange(4),
        "c": np.arange(3),
        "a": np.arange(3),
        "z": np.arange(5),
        "b": np.arange(2),
        "y": np.arange(4),
        "f": np.arange(2),
    }

    coord_array = coordarray(data, coords)
    coord_array2 = coordarray(data2, coords2)

    res = coord_array + coord_array2

    assert res.shape == (3, 4, 5, 3, 2, 3, 4, 5, 2)
    assert res.axes == ("x", "y", "z", "a", "b", "c", "d", "e", "f")

    res = -coord_array * coord_array2

    assert res.shape == (3, 4, 5, 3, 2, 3, 4, 5, 2)
    assert res.axes == ("x", "y", "z", "a", "b", "c", "d", "e", "f")


def test_axes_indexing():
    """Test axes indexing."""
    data = np.arange(0, 100).reshape(10, 10)
    coords = {"x": np.arange(10), "y": np.arange(10)}
    coord_array = coordarray(data, coords)

    np.testing.assert_array_equal(coord_array["x", -1], coord_array["x", 9])


def test_can_become_quantity_inplace():
    """Test if coord array can become quantity."""
    data = np.arange(10) + 1
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)

    coord_array *= u.K

    assert coord_array.shape == (10,)
    assert coord_array.ndim == 1
    assert coord_array.size == 10
    assert coord_array.data.shape == (10,)
    assert coord_array.data.unit == u.K

    coord_array *= u.m

    assert coord_array.shape == (10,)
    assert coord_array.data.unit == u.K * u.m

    coord_array /= u.m

    assert coord_array.shape == (10,)
    assert coord_array.data.unit == u.K


def test_can_become_quantity():
    """Test if coord array can become quantity."""
    data = np.arange(10) + 1
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)

    new_coord_array = coord_array << u.K

    assert new_coord_array.shape == (10,)
    assert new_coord_array.ndim == 1
    assert new_coord_array.size == 10
    assert new_coord_array.value.shape == (10,)
    assert new_coord_array.value.unit == u.K

    new_coord_array = coord_array * u.m

    assert new_coord_array.shape == (10,)
    assert new_coord_array.value.unit == u.m

    new_coord_array = coord_array / u.Pa

    assert new_coord_array.shape == (10,)
    assert new_coord_array.value.unit == 1 / u.Pa


def test_can_become_quantity_left():
    """Test if coord array can become quantity."""
    data = np.arange(10) + 1
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)

    new_coord_array = u.K * coord_array

    assert new_coord_array.shape == (10,)
    assert new_coord_array.ndim == 1
    assert new_coord_array.size == 10
    assert new_coord_array.data.shape == (10,)
    assert new_coord_array.unit == u.K

    new_coord_array = u.m * coord_array

    assert new_coord_array.shape == (10,)
    assert new_coord_array.unit == u.m

    new_coord_array = 1 / u.Pa * coord_array

    assert new_coord_array.shape == (10,)
    assert new_coord_array.unit == 1 / u.Pa


def test_coordarray_ufuncs_nparray():
    """Test ufuncs for coord array."""
    data = np.arange(10)
    coords = {"x": np.arange(10)}
    coord_array = coordarray(data, coords)

    assert isinstance(np.exp(coord_array), coordarray)
    assert np.exp(coord_array).coords == coord_array.coords
    np.testing.assert_array_equal(-coord_array, -data)
    np.testing.assert_array_equal(np.sin(coord_array), np.sin(data))
    np.testing.assert_array_equal(np.cos(coord_array), np.cos(data))
    np.testing.assert_array_equal(np.tan(coord_array), np.tan(data))
    np.testing.assert_array_equal(np.arcsin(coord_array), np.arcsin(data))
    np.testing.assert_array_equal(np.arccos(coord_array), np.arccos(data))
    np.testing.assert_array_equal(np.arctan(coord_array), np.arctan(data))


def test_transpose_nparray():
    """Test transposition of coord array."""
    data = np.random.rand(20, 10)
    coords = {"x": np.arange(20), "y": np.arange(10)}

    coord_array = coordarray(data, coords)

    transposed = coord_array.transpose()
    assert transposed.shape == (10, 20)
    assert transposed.axes == ("y", "x")
    np.testing.assert_array_equal(data.T, transposed.data)

    transposed = coord_array.transpose("y", "x")
    assert transposed.shape == (10, 20)
    assert transposed.axes == ("y", "x")
    np.testing.assert_array_equal(data.T, transposed.data)

    transposed = coord_array.transpose("y", "x", "z")
    assert transposed.shape == (10, 20)
    assert transposed.axes == ("y", "x")
    np.testing.assert_array_equal(data.T, transposed.data)

    with pytest.raises(ValueError):
        transposed = coord_array.transpose("y", "x", "z", mode="raise")

    data = np.random.rand(20, 10, 15)
    coords = {"x": np.arange(20), "y": np.arange(10), "z": np.arange(15)}

    coord_array = coordarray(data, coords)

    transposed = coord_array.transpose()

    assert transposed.shape == (15, 10, 20)
    assert transposed.axes == ("z", "y", "x")
    np.testing.assert_array_equal(data.T, transposed.data)

    transposed = coord_array.transpose("y", "z", "x")

    assert transposed.shape == (10, 15, 20)
    assert transposed.axes == ("y", "z", "x")
    np.testing.assert_array_equal(data.transpose(1, 2, 0), transposed.data)


def test_operator_coordarray_diff():
    """Test test multiplication of coord array."""
    data = np.random.rand(20, 10)
    coords = {"x": np.arange(20), "y": np.arange(10)}

    coord_array = coordarray(data, coords)
    transposed = coord_array.transpose()

    assert coord_array.shape == (20, 10)
    assert transposed.shape == (10, 20)

    result = coord_array - transposed

    assert result.shape == (20, 10)
    assert result.axes == ("x", "y")


def test_slicing_np_array():
    """Test numpy style slicing."""
    data = np.random.rand(20, 10)
    coords = {"x": np.arange(20), "y": np.arange(10)}

    coord_array = coordarray(data, coords)

    assert coord_array[0].shape == (10,)
    assert coord_array[0].coords["x"] == 0


def test_coordarray_interp_match():
    """Test coordarray matching coordinates with interpolation."""
    data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [90, 100, 110]])
    coords = {"x": np.array([1.0, 2.0, 3.0, 4.0]), "y": np.array([10.0, 20.0, 30.0])}
    arr = coordarray(data, **coords)

    new_coords = {"x": np.array([1.5, 2.5, 2.7, 2.9]), "y": np.array([10.5, 20.5, 21, 25, 29])}

    interp_arr = arr.interp_match(new_coords)

    assert interp_arr.shape == (4, 5)
    assert interp_arr.axes == ("x", "y")
