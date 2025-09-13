"""Test that the 1D mesh and manifold works as would be expected."""

from interplib._interp import GeoID, Manifold, Manifold1D


def test_manifold1d():
    """Check that Manifold works as expected."""
    caught = False
    try:
        _ = Manifold()
    except TypeError:
        caught = True
    assert caught, "How did you construct the object?"

    caught = False
    try:
        _ = Manifold1D()
    except TypeError:
        caught = True
    assert caught, "How did you construct the object?"

    n = 10
    real_one = Manifold1D.line_mesh(n)
    assert real_one.n_lines == n
    assert real_one.n_points == n + 1
    for i in range(n):
        ln = real_one.get_line(i + 1)
        i_ln = real_one.find_line(ln)
        assert GeoID(i, 0) == i_ln
        ln2 = real_one.get_line(-(i + 1))
        assert ln.begin == ln2.end
        assert ln.end == ln2.begin
        assert ln.begin == GeoID(i, 0)
        assert ln.end == GeoID(i + 1, 0)

    caught = False
    try:
        _ = real_one.get_line(n + 1)
    except IndexError:
        caught = True
    assert caught, "How did you get a line that far off?"
