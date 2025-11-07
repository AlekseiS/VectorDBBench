"""Wrapper around the Milvus vector database over VectorDB"""

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, utility

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import MilvusIndexConfig

log = logging.getLogger(__name__)

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024


class Milvus(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MilvusIndexConfig,
        collection_name: str = "VDBBench",
        drop_old: bool = False,
        name: str = "Milvus",
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.batch_size = int(MILVUS_LOAD_REQS_SIZE / (dim * 4))
        self.with_scalar_labels = with_scalar_labels

        self._primary_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"

        from pymilvus import connections

        connections.connect(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            timeout=30,
        )
        if drop_old and utility.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        if not utility.has_collection(self.collection_name):
            fields = [
                # Use VARCHAR for string IDs, INT64 for integer IDs
                # You can make this configurable via db_case_config if needed
                FieldSchema(self._primary_field, DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim),
            ]
            if self.with_scalar_labels:
                is_partition_key = db_case_config.use_partition_key
                log.info(f"with_scalar_labels, add a new varchar field, as partition_key: {is_partition_key}")
                fields.append(
                    FieldSchema(
                        self._scalar_label_field,
                        DataType.VARCHAR,
                        max_length=256,
                        is_partition_key=is_partition_key,
                    )
                )

            log.info(f"{self.name} create collection: {self.collection_name}")

            # Create the collection
            col = Collection(
                name=self.collection_name,
                schema=CollectionSchema(fields),
                consistency_level="Session",
                num_shards=self.db_config.get("num_shards", 1),
            )

            self.create_index()
            col.load(replica_number=self.db_config.get("replica_number", 1))

        connections.disconnect("default")

    def create_index(self):
        col = Collection(self.collection_name)
        # vector index
        col.create_index(
            self._vector_field,
            self.case_config.index_param(),
            index_name=self._vector_index_name,
        )
        # scalar index for varchar (label-filter)
        if self.with_scalar_labels:
            col.create_index(
                self._scalar_label_field,
                index_params={
                    "index_type": "BITMAP",
                },
                index_name=self._scalar_labels_index_name,
            )

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        from pymilvus import connections

        self.col: Collection | None = None

        connections.connect(**self.db_config, timeout=60)
        # Grab the existing colection with connections
        self.col = Collection(self.collection_name)

        try:
            yield
        finally:
            connections.disconnect("default")
            self.col = None  # Clean up Collection object to avoid pickling issues

    def _optimize(self):
        log.info(f"{self.name} optimizing before search")
        self._post_insert()
        try:
            self.col.load(refresh=True)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def _post_insert(self):
        try:
            self.col.flush()
            # wait for index done and load refresh
            self.create_index()

            utility.wait_for_index_building_complete(self.collection_name, index_name=self._vector_index_name)

            def wait_index():
                while True:
                    progress = utility.index_building_progress(self.collection_name, index_name=self._vector_index_name)
                    if progress.get("pending_index_rows", -1) == 0:
                        break
                    time.sleep(5)

            wait_index()

            # Skip compaction if use GPU indexType
            if self.case_config.is_gpu_index:
                log.debug("skip compaction for gpu index type.")
            else:
                try:
                    self.col.compact()
                    self.col.wait_for_compaction_completed()
                    log.info("compactation completed. waiting for the rest of index buliding.")
                except Exception as e:
                    log.warning(f"{self.name} compact error: {e}")
                    if hasattr(e, "code"):
                        if e.code().name == "PERMISSION_DENIED":
                            log.warning("Skip compact due to permission denied.")
                    else:
                        raise e from e
                wait_index()
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        assert self.col, "Please call self.init() before"
        self._optimize()

    def rebuild_index(self):
        """Drop and rebuild index without touching data"""
        assert self.col, "Please call self.init() before"
        log.info(f"{self.name} rebuilding index on existing data")

        # Release collection before dropping index
        log.info(f"{self.name} releasing collection")
        self.col.release()

        # Drop existing indexes
        # Try to drop all indexes by field name
        try:
            log.info(f"{self.name} dropping vector index: {self._vector_index_name}")
            # Use index_name parameter instead of field_name
            self.col.drop_index(index_name=self._vector_index_name)
        except Exception as e:
            log.warning(f"{self.name} no vector index to drop or error: {e}")

        # Drop scalar index if it exists
        if self.with_scalar_labels:
            try:
                log.info(f"{self.name} dropping scalar index on field: {self._scalar_label_field}")
                # For scalar index, try dropping by field name since we don't have a named index
                indexes = self.col.indexes
                for idx in indexes:
                    if idx.field_name == self._scalar_label_field:
                        log.info(f"{self.name} found scalar index to drop")
                        self.col.drop_index(index_name=idx.index_name)
                        break
            except Exception as e:
                log.warning(f"{self.name} no scalar index to drop or error: {e}")

        # Rebuild index and load collection
        log.info(f"{self.name} creating new index")
        self._post_insert()

        # Load collection (not with refresh since it was released)
        log.info(f"{self.name} loading collection")
        self.col.load()

        log.info(f"{self.name} index rebuilt successfully")

        # Note: Collection will be disconnected and cleaned up by the context manager
        # The context manager's __exit__ will call connections.disconnect("default")

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        if self.case_config.is_gpu_index:
            log.info("current gpu_index only supports IP / L2, cosine dataset need normalize.")
            return True

        return False

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int | str],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into Milvus. should call self.init() first"""
        # use the first insert_embeddings to init collection
        assert self.col is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_data = [
                    metadata[batch_start_offset:batch_end_offset],
                    metadata[batch_start_offset:batch_end_offset],
                    embeddings[batch_start_offset:batch_end_offset],
                ]
                if self.with_scalar_labels:
                    insert_data.append(labels_data[batch_start_offset:batch_end_offset])
                res = self.col.insert(insert_data)
                insert_count += len(res.primary_keys)
        except MilvusException as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"{self._scalar_label_field} == '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Milvus - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int | str]:
        """Perform a search on a query embedding and return results."""
        assert self.col is not None

        # Perform the search.
        res = self.col.search(
            data=[query],
            anns_field=self._vector_field,
            param=self.case_config.search_param(),
            limit=k,
            expr=self.expr,
        )

        # Organize results (keep whatever type Milvus returns: int or str).
        return [result.id for result in res[0]]
