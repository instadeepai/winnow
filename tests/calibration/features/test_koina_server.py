"""Tests for targeting a self-hosted Koina server from the Koina-backed features."""

import pytest

from winnow.calibration.features.chimeric import ChimericFeatures
from winnow.calibration.features.constants import DEFAULT_KOINA_SERVER_URL
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.calibration.features.retention_time import RetentionTimeFeature


def _build(feature_class, **kwargs):
    """Construct a feature, supplying whichever required arguments it has."""
    if feature_class is RetentionTimeFeature:
        return feature_class(**kwargs)
    return feature_class(mz_tolerance=0.02, mz_tolerance_unit="da", **kwargs)


KOINA_FEATURES = [FragmentMatchFeatures, ChimericFeatures, RetentionTimeFeature]


def test_default_server_url_is_the_public_endpoint():
    """The fallback used when no server is configured is the public Koina endpoint."""
    assert DEFAULT_KOINA_SERVER_URL == "koina.wilhelmlab.org:443"


@pytest.mark.parametrize("feature_class", KOINA_FEATURES)
def test_server_is_unset_by_default(feature_class):
    """Without configuration a feature carries no server, so the public one is used."""
    feature = _build(feature_class)

    assert feature.koina_server_url is None
    assert feature.koina_ssl is True


@pytest.mark.parametrize("feature_class", KOINA_FEATURES)
def test_self_hosted_server_is_stored(feature_class):
    """A self-hosted server and its TLS setting are kept for the Koina client."""
    feature = _build(feature_class, koina_server_url="localhost:8500", koina_ssl=False)

    assert feature.koina_server_url == "localhost:8500"
    assert feature.koina_ssl is False


@pytest.mark.parametrize("feature_class", KOINA_FEATURES)
def test_server_can_be_overridden_after_construction(feature_class):
    """Features restored from a checkpoint are retargeted by assignment."""
    feature = _build(feature_class)

    feature.koina_server_url = "localhost:8500"
    feature.koina_ssl = False

    assert feature.koina_server_url == "localhost:8500"
    assert feature.koina_ssl is False
