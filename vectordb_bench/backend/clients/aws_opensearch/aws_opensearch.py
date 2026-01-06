import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from opensearchpy import OpenSearch

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AWSOpenSearchIndexConfig, AWSOS_Engine

log = logging.getLogger(__name__)

WAITING_FOR_REFRESH_SEC = 30
WAITING_FOR_FORCE_MERGE_SEC = 30
SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC = 30


class AWSOpenSearch(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AWSOpenSearchIndexConfig,
        index_name: str = "vdb_bench_index",  # must be lowercase
        id_col_name: str = "_id",
        label_col_name: str = "label",
        vector_col_name: str = "embedding",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_name = index_name
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_col_name = vector_col_name
        self.with_scalar_labels = with_scalar_labels

        log.info(f"AWS_OpenSearch client config: {self.db_config}")
        log.info(f"AWS_OpenSearch db case config : {self.case_config}")
        client = OpenSearch(**self.db_config)
        if drop_old:
            log.info(f"AWS_OpenSearch client drop old index: {self.index_name}")
            is_existed = client.indices.exists(index=self.index_name)
            if is_existed:
                client.indices.delete(index=self.index_name)
            self._create_index(client)
        else:
            is_existed = client.indices.exists(index=self.index_name)
            if not is_existed:
                self._create_index(client)
                log.info(f"AWS_OpenSearch client create index: {self.index_name}")

            self._configure_cluster_settings(client)
            self._update_ef_search_before_search(client)
            self._load_graphs_to_memory(client)

    def _create_index(self, client: OpenSearch) -> None:
        self._log_index_creation_info()
        self._configure_cluster_settings(client)
        settings = self._build_index_settings()
        vector_field_config = self._build_vector_field_config()
        mappings = self._build_mappings(vector_field_config)
        self._create_opensearch_index(client, settings, mappings)

    def _log_index_creation_info(self) -> None:
        log.info(f"Creating index with ef_search: {self.case_config.ef_search}")
        log.info(f"Creating index with number_of_replicas: {self.case_config.number_of_replicas}")
        log.info(f"Creating index with engine: {self.case_config.engine}")
        log.info(f"Creating index with metric type: {self.case_config.metric_type_name}")
        log.info(f"All case_config parameters: {self.case_config.__dict__}")

    def _configure_cluster_settings(self, client: OpenSearch) -> None:
        cluster_settings_body = {
            "persistent": {
                "knn.algo_param.index_thread_qty": self.case_config.index_thread_qty,
                "knn.memory.circuit_breaker.limit": self.case_config.cb_threshold,
            }
        }
        client.cluster.put_settings(body=cluster_settings_body)

    def _build_index_settings(self) -> dict:
        return {
            "index": {
                "knn": True,
                "number_of_shards": self.case_config.number_of_shards,
                "number_of_replicas": self.case_config.number_of_replicas,
                "translog.flush_threshold_size": self.case_config.flush_threshold_size,
                "knn.advanced.approximate_threshold": "10000",
                "knn.algo_param.ef_search": self.case_config.ef_search,
            },
            "refresh_interval": self.case_config.refresh_interval,
        }

    def _build_vector_field_config(self) -> dict:
        method_config = self.case_config.index_param()
        log.info(f"Raw method config from index_param(): {method_config}")

        if self.case_config.engine == AWSOS_Engine.s3vector:
            method_config = {"engine": "s3vector"}

        if self.case_config.on_disk:
            space_type = self.case_config.parse_metric()
            vector_field_config = {
                "type": "knn_vector",
                "dimension": self.dim,
                "space_type": space_type,
                "data_type": "float",
                "mode": "on_disk",
                "compression_level": "32x",
            }
            log.info("Using on-disk vector configuration with compression_level: 32x")
        else:
            vector_field_config = {
                "type": "knn_vector",
                "dimension": self.dim,
                "method": method_config,
            }

        if self.case_config.on_disk:
            log.info(f"Final on-disk vector field config: {vector_field_config}")
        elif self.case_config.engine == AWSOS_Engine.s3vector:
            space_type = self.case_config.parse_metric()
            vector_field_config["space_type"] = space_type
            vector_field_config["method"] = {"engine": "s3vector"}
            log.info(f"Final vector field config for s3vector: {vector_field_config}")
        else:
            log.info(f"Standard vector field config: {vector_field_config}")

        return vector_field_config

    def _build_mappings(self, vector_field_config: dict) -> dict:
        if self.case_config.engine == AWSOS_Engine.s3vector:
            mappings = {
                "properties": {
                    self.label_col_name: {"type": "keyword"},
                    self.vector_col_name: vector_field_config,
                },
            }
            log.info("Using simplified mappings for s3vector engine (no _source configuration)")
        else:
            mappings = {
                "_source": {"excludes": [self.vector_col_name], "recovery_source_excludes": [self.vector_col_name]},
                "properties": {
                    self.label_col_name: {"type": "keyword"},
                    self.vector_col_name: vector_field_config,
                },
            }
            log.info("Using standard mappings with _source configuration for non-s3vector engines")
        return mappings

    def _create_opensearch_index(self, client: OpenSearch, settings: dict, mappings: dict) -> None:
        try:
            log.info(f"Creating index with settings: {settings}")
            log.info(f"Creating index with mappings: {mappings}")

            if self.case_config.engine == AWSOS_Engine.s3vector:
                method_in_mappings = mappings["properties"][self.vector_col_name]["method"]
                log.info(f"Final method config being sent to OpenSearch: {method_in_mappings}")

            client.indices.create(
                index=self.index_name,
                body={"settings": settings, "mappings": mappings},
            )

            if self.case_config.engine == AWSOS_Engine.s3vector:
                self._verify_s3vector_index_config(client)

        except Exception as e:
            log.warning(f"Failed to create index: {self.index_name} error: {e!s}")
            raise e from None

    def _verify_s3vector_index_config(self, client: OpenSearch) -> None:
        try:
            actual_mapping = client.indices.get_mapping(index=self.index_name)
            actual_method = actual_mapping[self.index_name]["mappings"]["properties"][self.vector_col_name]["method"]
            log.info(f"Actual method config in created index: {actual_method}")
        except Exception as e:
            log.warning(f"Failed to verify index configuration: {e}")

    @contextmanager
    def init(self) -> None:
        """connect to opensearch"""
        self.client = OpenSearch(**self.db_config)

        yield
        self.client = None
        del self.client

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert the embeddings to the opensearch."""
        assert self.client is not None, "should self.init() first"

        # 使用 number_of_indexing_clients 作为线程数
        num_clients = self.case_config.number_of_indexing_clients or 1
        if num_clients <= 1:
            log.info("Using single client for data insertion")
            return self._insert_with_single_client(embeddings, metadata, labels_data)

        log.info("Using %s parallel threads for data insertion", num_clients)
        return self._insert_with_multiple_clients(embeddings, metadata, num_clients, labels_data)

    def _insert_with_single_client(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
    ) -> tuple[int, Exception]:
        insert_data = []
        for i in range(len(embeddings)):
            index_data = {"index": {"_index": self.index_name, self.id_col_name: metadata[i]}}
            if self.with_scalar_labels and self.case_config.use_routing and labels_data is not None:
                index_data["routing"] = labels_data[i]
            insert_data.append(index_data)

            other_data = {self.vector_col_name: embeddings[i]}
            if self.with_scalar_labels and labels_data is not None:
                other_data[self.label_col_name] = labels_data[i]
            insert_data.append(other_data)

        try:
            self.client.bulk(body=insert_data)
            return len(embeddings), None
        except Exception as e:
            log.warning(f"Failed to insert data: {self.index_name} error: {e!s}")
            time.sleep(10)
            return self._insert_with_single_client(embeddings, metadata, labels_data)

    def _insert_with_multiple_clients(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        num_clients: int,
        labels_data: list[str] | None = None,
    ) -> tuple[int, Exception]:
        """Use multiple threads to insert data in streaming fashion.
        
        Reads data in batches and submits each batch to a thread immediately.
        If thread pool queue is full, waits for a thread to become available.
        This reduces memory usage and starts writing earlier.
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock, local

        # 获取批次大小配置
        batch_size = getattr(self.case_config, "bulk_docs_per_chunk", 2000)
        if batch_size <= 0:
            batch_size = 2000

        num_clients = max(1, num_clients)

        log.info(
            "Starting streaming multi-thread insert: num_threads=%s, batch_size=%s",
            num_clients,
            batch_size,
        )

        thread_state = local()
        clients: list[OpenSearch] = []
        clients_lock = Lock()

        def _get_thread_local_client() -> OpenSearch:
            client = getattr(thread_state, "client", None)
            if client is None:
                client = OpenSearch(**self.db_config)
                thread_state.client = client
                with clients_lock:
                    clients.append(client)
            return client

        def _build_bulk_payload(
            chunk_embeddings: list[list[float]],
            chunk_metadata: list[int],
            chunk_labels_data: list[str] | None,
        ) -> list[dict]:
            insert_data: list[dict] = []
            for i in range(len(chunk_embeddings)):
                index_data = {"index": {"_index": self.index_name, self.id_col_name: chunk_metadata[i]}}
                if self.with_scalar_labels and self.case_config.use_routing and chunk_labels_data is not None:
                    index_data["routing"] = chunk_labels_data[i]
                insert_data.append(index_data)

                other_data = {self.vector_col_name: chunk_embeddings[i]}
                if self.with_scalar_labels and chunk_labels_data is not None:
                    other_data[self.label_col_name] = chunk_labels_data[i]
                insert_data.append(other_data)
            return insert_data

        def insert_chunk(
            chunk_embeddings: list[list[float]],
            chunk_metadata: list[int],
            chunk_labels_data: list[str] | None,
        ) -> tuple[int, Exception]:
            client = _get_thread_local_client()
            insert_data = _build_bulk_payload(chunk_embeddings, chunk_metadata, chunk_labels_data)

            try:
                resp = client.bulk(body=insert_data)
                inserted_docs = len(resp.get("items", []))
                log.debug("Thread inserted %s documents in one bulk", inserted_docs)
                return len(chunk_embeddings), None
            except Exception as exc:
                log.warning("Thread failed to insert data: %s", exc)
                return 0, exc

        results: list[tuple[int, Exception]] = []
        active_futures: list[concurrent.futures.Future] = []  # 跟踪正在运行的任务
        all_futures: list[concurrent.futures.Future] = []  # 跟踪所有已提交的任务
        batch_idx = 0
        current_batch_embeddings: list[list[float]] = []
        current_batch_metadata: list[int] = []
        current_batch_labels: list[str] | None = [] if labels_data is not None else None
        current_idx = 0

        # 使用线程池，max_workers 控制并发线程数
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            # 主循环：持续读取数据并提交批次
            for emb in embeddings:
                current_batch_embeddings.append(emb)
                current_batch_metadata.append(metadata[current_idx])
                if labels_data is not None:
                    current_batch_labels.append(labels_data[current_idx])
                current_idx += 1

                # 当批次达到指定大小时，提交给线程池
                if len(current_batch_embeddings) >= batch_size:
                    # 如果线程池满了（活跃任务数达到 max_workers），等待一个任务完成
                    while len(active_futures) >= num_clients:
                        # 等待至少一个任务完成
                        done, not_done = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        # 收集已完成任务的结果
                        for future in done:
                            try:
                                result = future.result()
                                results.append(result)
                                log.debug("Batch completed: %s documents", result[0])
                            except Exception as exc:
                                results.append((0, exc))
                                log.warning("Batch failed: %s", exc)
                            active_futures.remove(future)
                    
                    # 构建任务参数
                    task_embeddings = current_batch_embeddings.copy()
                    task_metadata = current_batch_metadata.copy()
                    task_labels = current_batch_labels.copy() if current_batch_labels is not None else None
                    
                    # 提交任务（此时线程池肯定有空闲，不会阻塞）
                    future = executor.submit(insert_chunk, task_embeddings, task_metadata, task_labels)
                    active_futures.append(future)
                    all_futures.append(future)
                    batch_idx += 1
                    
                    log.debug("Submitted batch %s with %s documents (active threads: %s/%s)", 
                             batch_idx, len(current_batch_embeddings), len(active_futures), num_clients)
                    
                    # 清空当前批次，继续读取下一批
                    current_batch_embeddings.clear()
                    current_batch_metadata.clear()
                    if current_batch_labels is not None:
                        current_batch_labels.clear()

            # 提交最后一个不完整的批次（如果有）
            if current_batch_embeddings:
                # 如果线程池满了，等待一个任务完成
                while len(active_futures) >= num_clients:
                    done, not_done = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        try:
                            result = future.result()
                            results.append(result)
                            log.debug("Batch completed: %s documents", result[0])
                        except Exception as exc:
                            results.append((0, exc))
                            log.warning("Batch failed: %s", exc)
                        active_futures.remove(future)
                
                future = executor.submit(insert_chunk, current_batch_embeddings, current_batch_metadata, current_batch_labels)
                active_futures.append(future)
                all_futures.append(future)
                log.debug("Submitted final batch %s with %s documents", batch_idx + 1, len(current_batch_embeddings))

            # 读取完成，等待所有剩余任务完成
            log.info("Reading complete. Waiting for %s remaining batches to finish", len(active_futures))
            for future in concurrent.futures.as_completed(active_futures):
                try:
                    result = future.result()
                    results.append(result)
                    log.debug("Batch completed: %s documents", result[0])
                except Exception as exc:
                    results.append((0, exc))
                    log.warning("Batch failed: %s", exc)

        # 清理客户端连接
        from contextlib import suppress

        for client in clients:
            with suppress(Exception):
                client.close()

        total_count = sum(count for count, _ in results)
        errors = [err for _, err in results if err is not None]

        if errors:
            log.warning(
                "Multi-thread insert encountered %s error(s). "
                "Successfully inserted %s documents. First error: %s",
                len(errors),
                total_count,
                errors[0],
            )
            # 返回已成功插入的数量和第一个错误
            return total_count, errors[0]

        # 使用主进程已有 client 查询统计信息
        try:
            resp = self.client.indices.stats(index=self.index_name)
            total_indexed = resp["_all"]["primaries"]["indexing"]["index_total"]
            log.info(
                "Streaming multi-thread insertion complete. Total documents indexed in index stats: %s (this batch: %s)",
                total_indexed,
                total_count,
            )
        except Exception as exc:
            log.warning("Failed to fetch index stats after multi-thread insert: %s", exc)

        return total_count, None

    def _update_ef_search_before_search(self, client: OpenSearch):
        ef_search_value = self.case_config.ef_search
        try:
            index_settings = client.indices.get_settings(index=self.index_name)
            current_ef_search = (
                index_settings.get(self.index_name, {})
                .get("settings", {})
                .get("index", {})
                .get("knn.algo_param", {})
                .get("ef_search")
            )

            if current_ef_search != str(ef_search_value):
                log.info(f"Updating ef_search before search from {current_ef_search} to {ef_search_value}")
                settings_body = {"index": {"knn.algo_param.ef_search": ef_search_value}}
                client.indices.put_settings(index=self.index_name, body=settings_body)
                log.info(f"Successfully updated ef_search to {ef_search_value} before search")

            log.info(f"Current engine: {self.case_config.engine}")
            log.info(f"Current metric_type: {self.case_config.metric_type_name}")

        except Exception as e:
            log.warning(f"Failed to update ef_search parameter before search: {e}")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.

        Returns:
            list[int]: list of k most similar ids to the query embedding.
        """
        assert self.client is not None, "should self.init() first"

        # Configure query based on engine type
        if self.case_config.engine == AWSOS_Engine.s3vector:
            # For s3vector engine, use simplified query without method_parameters
            knn_query = {
                "vector": query,
                "k": k,
                **({"filter": self.filter} if self.filter else {}),
            }
            log.debug("Using simplified knn query for s3vector engine (no method_parameters)")
        else:
            # For other engines (faiss, lucene), use standard query with method_parameters
            knn_query = {
                "vector": query,
                "k": k,
                "method_parameters": self.case_config.search_param(),
                **({"filter": self.filter} if self.filter else {}),
                "rescore": {"oversample_factor": self.case_config.oversample_factor}
                # if self.case_config.use_quant
                # else {}
                ,
            }
            log.debug("Using standard knn query with method_parameters for non-s3vector engines")

        body = {
            "size": k,
            "query": {"knn": {self.vector_col_name: knn_query}},
        }

        try:
            resp = self.client.search(
                index=self.index_name,
                body=body,
                size=k,
                _source=False,
                docvalue_fields=[self.id_col_name],
                stored_fields="_none_",
                preference="_only_local" if self.case_config.number_of_shards == 1 else None,
                routing=self.routing_key,
            )
            log.debug(f"Search took: {resp['took']}")
            log.debug(f"Search shards: {resp['_shards']}")
            log.debug(f"Search hits total: {resp['hits']['total']}")
            try:
                return [int(h["fields"][self.id_col_name][0]) for h in resp["hits"]["hits"]]
            except Exception:
                # empty results
                return []
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise e from None

    def prepare_filter(self, filters: Filter):
        self.routing_key = None
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            self.filter = {"range": {self.id_col_name: {"gt": filters.int_value}}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {"term": {self.label_col_name: filters.label_value}}
            if self.case_config.use_routing:
                self.routing_key = filters.label_value
        else:
            msg = f"Not support Filter for OpenSearch - {filters}"
            raise ValueError(msg)

    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases."""
        self._update_ef_search()
        # Call refresh first to ensure that all segments are created
        self._refresh_index()
        if self.case_config.force_merge_enabled:
            self._do_force_merge()
            self._refresh_index()
        self._update_replicas()
        # Call refresh again to ensure that the index is ready after force merge.
        self._refresh_index()
        # ensure that all graphs are loaded in memory and ready for search
        self._load_graphs_to_memory(self.client)

    def _update_ef_search(self):
        ef_search_value = (
            self.case_config.ef_search if self.case_config.ef_search is not None else self.case_config.efSearch
        )
        log.info(f"Updating ef_search parameter to: {ef_search_value}")

        settings_body = {"index": {"knn.algo_param.ef_search": ef_search_value}}
        try:
            self.client.indices.put_settings(index=self.index_name, body=settings_body)
            log.info(f"Successfully updated ef_search to {ef_search_value}")
            log.info(f"Current engine: {self.case_config.engine}")
            log.info(f"Current metric_type: {self.case_config.metric_type}")
        except Exception as e:
            log.warning(f"Failed to update ef_search parameter: {e}")

    def _update_replicas(self):
        index_settings = self.client.indices.get_settings(index=self.index_name)
        current_number_of_replicas = int(index_settings[self.index_name]["settings"]["index"]["number_of_replicas"])
        log.info(
            f"Current Number of replicas are {current_number_of_replicas}"
            f" and changing the replicas to {self.case_config.number_of_replicas}"
        )
        settings_body = {"index": {"number_of_replicas": self.case_config.number_of_replicas}}
        self.client.indices.put_settings(index=self.index_name, body=settings_body)
        self._wait_till_green()

    def _wait_till_green(self):
        log.info("Wait for index to become green..")
        while True:
            res = self.client.cat.indices(index=self.index_name, h="health", format="json")
            health = res[0]["health"]
            if health == "green":
                break
            log.info(f"The index {self.index_name} has health : {health} and is not green. Retrying")
            time.sleep(SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC)
        log.info(f"Index {self.index_name} is green..")

    def _refresh_index(self):
        log.debug(f"Starting refresh for index {self.index_name}")
        while True:
            try:
                log.info("Starting the Refresh Index..")
                self.client.indices.refresh(index=self.index_name)
                break
            except Exception as e:
                log.info(
                    f"Refresh errored out. Sleeping for {WAITING_FOR_REFRESH_SEC} sec and then Retrying : {e}",
                )
                time.sleep(WAITING_FOR_REFRESH_SEC)
                continue
        log.debug(f"Completed refresh for index {self.index_name}")

    def _do_force_merge(self):
        log.info(f"Updating the Index thread qty to {self.case_config.index_thread_qty_during_force_merge}.")

        cluster_settings_body = {
            "persistent": {"knn.algo_param.index_thread_qty": self.case_config.index_thread_qty_during_force_merge}
        }
        self.client.cluster.put_settings(body=cluster_settings_body)

        # log.info("Updating the graph threshold to ensure that during merge we can do graph creation.")
        # output = self.client.indices.put_settings(
        #     index=self.index_name, body={"index.knn.advanced.approximate_threshold": "0"}
        # )
        # log.info(f"response of updating setting is: {output}")

        log.info(f"Starting force merge for index {self.index_name}")
        segments = self.case_config.number_of_segments
        force_merge_endpoint = f"/{self.index_name}/_forcemerge?max_num_segments={segments}&wait_for_completion=false"
        force_merge_task_id = self.client.transport.perform_request("POST", force_merge_endpoint)["task"]
        while True:
            time.sleep(WAITING_FOR_FORCE_MERGE_SEC)
            task_status = self.client.tasks.get(task_id=force_merge_task_id)
            if task_status["completed"]:
                break
        log.info(f"Completed force merge for index {self.index_name}")

    def _load_graphs_to_memory(self, client: OpenSearch):
        if self.case_config.engine != AWSOS_Engine.lucene:
            log.info("Calling warmup API to load graphs into memory")
            warmup_endpoint = f"/_plugins/_knn/warmup/{self.index_name}"
            client.transport.perform_request("GET", warmup_endpoint)
