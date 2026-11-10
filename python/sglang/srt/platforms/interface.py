"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, capability flags, and
configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.srt.platforms" group and should subclass SRTPlatform.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional, Type

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# Re-export for convenience
__all__ = ["SRTPlatform", "PlatformEnum"]


class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.

    Inherits device identity queries and operations from DeviceMixin.
    Adds SRT-specific factory methods, capability flags, and lifecycle hooks.

    OOT platforms should subclass SRTPlatform and override the methods
    relevant to their hardware.
    """

    # SRT-specific class-level attribute
    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific default values to server arguments.

        Called after ServerArgs is parsed.
        """
        pass

    def validate_input_token_ids(self, input_ids, vocabulary_size: int) -> None:
        """Validate host request tokens before they are sent to the device."""
        pass

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        """Return the default attention backend name for this platform."""
        raise NotImplementedError

    def get_graph_runner_cls(self) -> type:
        """Return the graph runner class for this platform."""
        raise NotImplementedError

    def get_mha_kv_pool_cls(self) -> type:
        """Return the MHA KV pool class for this platform."""
        raise NotImplementedError

    def get_mla_kv_pool_cls(self) -> type:
        """Return the MLA KV pool class for this platform."""
        raise NotImplementedError

    def get_dsa_kv_pool_cls(self) -> type:
        """Return the DSA KV pool class for this platform (DeepSeek V3.2)."""
        raise NotImplementedError

    def get_paged_allocator_cls(self) -> type:
        """Return the paged allocator class for this platform."""
        raise NotImplementedError

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the compilation backend identifier.

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        """
        return "inductor"

    def get_piecewise_backend_cls(self) -> type:
        """Return the piecewise compilation backend class for this platform."""
        raise NotImplementedError

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:
        """Return hardware-specific quantization config for the specific
        quantization scheme, raise an error if not supported or return None
        to use the default config."""
        return None

    # ------------------------------------------------------------------
    # Capability flags (safe conservative defaults)
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        """Whether this platform supports FP8 quantization."""
        return False

    def supports_compressed_w8a8_fp8(self, weight_quant, input_quant) -> Optional[bool]:
        """Return an explicit FP8 scheme decision, or None to use the default device check."""
        return None

    def prepare_w8a8_fp8_weight(self, layer, strategy, scale_name: str) -> bool:
        """Prepare a loaded FP8 linear weight if this platform owns its layout."""
        return False

    def get_w8a8_block_fp8_linear(self):
        """Return a platform block-FP8 linear implementation, if one is supplied."""
        return None

    def get_w8a8_fp8_linear(self):
        """Return a platform per-channel or per-tensor FP8 linear implementation."""
        return None

    def support_cuda_graph(self) -> bool:
        """Whether this platform supports device graph capture and replay.
        Controls CUDA graph (CudaGraphRunner) for the decode path.
        OOT platforms that support graph-style capture should return True.
        """
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """Whether this platform supports piecewise CUDA graph.

        Controls PiecewiseCudaGraphRunner for the prefill/extend path
        (torch.compile backend).
        """
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization.  Called in each worker."""
        pass

    def prepare_model_forward(self, forward):
        """Prepare a model forward callable for this platform."""
        return forward

    def requires_graceful_worker_shutdown(self):
        """Whether engine workers must complete their own process teardown."""
        return False

    def rope_cache_initialization_context(self):
        """Scope platform-specific CPU preparation of rotary lookup tables."""
        return nullcontext()

    def rope_cache_computation_device(self, cache_device):
        """Return the device used to compute additional rotary lookup rows."""
        return cache_device

    def paged_decode_last_locations(self, pool, request_indices, sequence_lengths):
        """Return prior device KV locations, or None to use normal indexing."""
        return None

    def prepare_rope_cache(self, cache):
        """Prepare a rotary lookup table for this platform."""
        return cache

    def attention_accepts_flat_kv(self) -> bool:
        """Whether the attention backend accepts K/V without splitting heads."""
        return False

    def get_last_token_indices(self):
        """Return prepared last-token indices, or None for the default computation."""
        return None

    def write_req_to_token_pool(
        self, pool, request_indices, prefix_lengths, extend_lengths, prefixes, locations,
        *, is_encoder_decoder=False,
    ) -> bool:
        """Write request mappings, or return False to use the default implementation."""
        return False

    def prepare_decode_positions(self, seq_lens_cpu, device):
        """Prepare decode positions, or return None for the default implementation."""
        return None

    def gather_scheduler_values(self, buffer, indices):
        """Gather from a request-pool buffer using allocator-owned indices."""
        return buffer[indices]

    def store_scheduler_values(self, buffer, indices, values):
        """Store to a request-pool buffer using allocator-owned indices."""
        buffer[indices] = values.to(buffer.dtype)

    def advance_decode_lengths(self, seq_lens, orig_seq_lens):
        """Return updated device lengths without changing prior batch tensors."""
        return seq_lens + 1, orig_seq_lens + 1

    def defer_eager_decode_positions(self) -> bool:
        """Whether host decode positions are prepared by the attention backend."""
        return False

    def copy_eager_inputs(self) -> bool:
        """Whether eager execution needs fixed-address input buffers."""
        return True

    def prepare_extend_positions(self, prefix_lens_cpu, seq_lens_cpu, total, device):
        """Prepare extend positions, or return None for the default implementation."""
        return None

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        """Return the dispatch key name for MultiPlatformOp.

        Determines which ``forward_<key>()`` method is selected.
        E.g. "cuda", "npu", "hip", "xpu", "cpu".
        """
        return "native"
