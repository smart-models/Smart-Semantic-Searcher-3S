"""Smoke tests for collection schema imports and point id determinism."""

from qdrant_middleware.core.config import get_settings
from qdrant_middleware.schemas.collections import CollectionReport, CollectionSchema
from qdrant_middleware.services.collection_schema import make_point_id


def test_collection_schema_models_import_from_package():
    schema = CollectionSchema.from_settings(get_settings())

    assert schema.dense_on_disk is not None
    assert schema.quantization is not None
    assert schema.sparse_idf is not None
    assert schema.sparse_on_disk is not None
    assert CollectionReport.model_fields["collection_name"] is not None


def test_make_point_id_is_stable():
    assert make_point_id("source", 1) == make_point_id("source", 1)
    assert make_point_id("source", 1) != make_point_id("source", 2)
