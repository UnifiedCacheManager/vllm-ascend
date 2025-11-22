# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from ucm.integration.vllm.v1_adapter import UCMConnector
logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class UCMConnectorV1(KVConnectorBase_V1):

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None

        ImplCls = UCMConnector
        self._engine = ImplCls(vllm_config, role, self)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        self._engine.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self._engine.wait_for_save()

    def clear_connector_metadata(self) -> None:
        self._engine.clear_connector_metadata()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self._engine.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        
        return self._engine.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        self._engine.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self._engine.build_connector_meta(scheduler_output)
    
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
         self._engine.bind_connector_metadata(connector_metadata)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._engine.request_finished(request, block_ids)
