from lazychemvis.helpers import cache


def test_signature_is_stable_and_order_sensitive(library_a, library_b):
    assert cache.library_signature(library_a) == cache.library_signature(
        list(library_a)
    )
    assert cache.library_signature(library_a) != cache.library_signature(library_b)
    assert cache.library_signature(library_a) != cache.library_signature(library_a[:-1])


def test_missing_key_never_matches(tmp_path, library_a):
    key = cache.cache_key(library_a, radius=2)
    assert not cache.matches(str(tmp_path), key)
    assert "no cache key" in cache.mismatch_reason(str(tmp_path), key)


def test_round_trip_and_mismatch_reasons(tmp_path, library_a, library_b):
    key = cache.cache_key(library_a, radius=2, descriptors=("a", "b"))
    cache.write_key(str(tmp_path), key)
    assert cache.matches(str(tmp_path), key)

    other_library = cache.cache_key(library_b, radius=2, descriptors=("a", "b"))
    assert (
        cache.mismatch_reason(str(tmp_path), other_library)
        == "the input library changed"
    )

    other_setting = cache.cache_key(library_a, radius=3, descriptors=("a", "b"))
    assert "radius" in cache.mismatch_reason(str(tmp_path), other_setting)
