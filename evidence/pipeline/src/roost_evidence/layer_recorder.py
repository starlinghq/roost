"""Hook capture on a transformers model (build plan R-011): the reference side of the per-layer stage.

Tool time only, like reference_runner: nothing here imports torch or safetensors at module level, and the model is
whatever `AutoModelForCausalLM` loaded. Forward hooks sit on the token embedding, on every decoder layer and on the
final norm; `output_hidden_states` is not used because its last entry is already normalised, not the last layer's
output. Decoder layers return a tensor in some versions and a tuple in others; the first element is taken.

A Gated DeltaNet layer's recurrent and convolution state is not in what its hook returns: it lives in the model's cache
object (`past_key_values.layers[i].recurrent_states[0]` and `.conv_states[0]` in the pinned transformers). The cache is
updated in place by the next forward, so each state is read after the forward that produced it and copied at once.
The layout of what is written is in layer_captures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roost_evidence import layer_captures


class CaptureSetupError(RuntimeError):
    """The model does not have the structure the recorder hooks (no decoder layer list, no cache states)."""


class LayerRecorder:
    """Records hidden states of every forward and, for the first `state_forwards` forwards, recurrent states."""

    def __init__(self, network: Any, *, state_forwards: int) -> None:
        if state_forwards < 0:
            raise ValueError("state_forwards must not be negative")
        self.state_forwards = state_forwards
        self._tensors: dict[str, Any] = {}
        self._current: dict[str, Any] = {}
        self._forward = 0
        self._hidden_size = 0
        self._handles: list[Any] = []

        config = network.config.get_text_config()
        count = int(config.num_hidden_layers)
        types = getattr(config, "layer_types", None)
        self.layer_types: list[str] = list(types) if types else ["full_attention"] * count
        if len(self.layer_types) != count:
            raise CaptureSetupError(f"config lists {len(self.layer_types)} layer types for {count} layers")

        container_name, layers = self._find_layers(network, count)
        parent = network.get_submodule(container_name.rpartition(".")[0]) if "." in container_name else network
        self.stages: list[str] = []
        embedding = network.get_input_embeddings()
        if embedding is not None:
            self._hook(embedding, layer_captures.EMBEDDINGS)
        for index, layer in enumerate(layers):
            self._hook(layer, layer_captures.layer_stage(index))
        norm = getattr(parent, "norm", None)
        if norm is not None and hasattr(norm, "register_forward_hook"):
            self._hook(norm, layer_captures.FINAL_NORM)

    @staticmethod
    def _find_layers(network: Any, count: int) -> tuple[str, Any]:
        for name, module in network.named_modules():
            if name.rpartition(".")[2] == "layers" and hasattr(module, "__len__") and len(module) == count:
                return name, module
        raise CaptureSetupError(f"no module list named 'layers' with {count} decoder layers was found")

    def _hook(self, module: Any, stage: str) -> None:
        self.stages.append(stage)

        def capture(_module: Any, _args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if tensor.dim() != 3 or tensor.shape[0] != 1:
                raise CaptureSetupError(f"{stage}: expected a [1, positions, hidden] output, got {list(tensor.shape)}")
            self._hidden_size = int(tensor.shape[-1])
            self._current[stage] = tensor[0].detach().to("cpu").float().clone()

        self._handles.append(module.register_forward_hook(capture))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def start_case(self) -> None:
        self._tensors = {}
        self._current = {}
        self._forward = 0

    def begin_forward(self) -> None:
        self._current = {}

    def end_forward(self, cache: Any) -> None:
        """Move this forward's hook captures into the case and copy the recurrent states out of `cache`."""
        missing = [s for s in self.stages if s not in self._current]
        if missing:
            raise CaptureSetupError(f"forward {self._forward}: no hook fired for {missing}")
        for stage in self.stages:
            self._tensors[layer_captures.hidden_name(self._forward, stage)] = self._current[stage]
        if self._forward < self.state_forwards:
            for index, kind in enumerate(self.layer_types):
                if kind == layer_captures.LINEAR:
                    self._copy_states(cache, index)
        self._forward += 1
        self._current = {}

    def _copy_states(self, cache: Any, index: int) -> None:
        layers = getattr(cache, "layers", None)
        if layers is None or index >= len(layers):
            raise CaptureSetupError("the cache object has no per-layer states; the pinned transformers is required")
        for kind in ("recurrent_state", "conv_state"):
            store = getattr(layers[index], f"{kind}s", None)
            state = store.get(0) if store else None
            if state is None:
                raise CaptureSetupError(f"cache layer {index} holds no {kind} after forward {self._forward}")
            self._tensors[layer_captures.state_name(self._forward, index, kind)] = (
                state[0].detach().to("cpu").float().clone()
            )

    def save(self, path: Path, prompt_ids: list[int], decode_ids: list[int]) -> str:
        """Write the case's capture as safetensors and return its sha256."""
        from safetensors.torch import save_file

        if self._forward != len(decode_ids) + 1:
            raise CaptureSetupError(f"{self._forward} forwards recorded for {len(decode_ids)} fed tokens")
        metadata = layer_captures.metadata_for(
            layer_types=self.layer_types,
            hidden_size=self._hidden_size,
            stages=self.stages_in_model_order(),
            prompt_ids=prompt_ids,
            decode_ids=decode_ids,
            state_forwards=min(self.state_forwards, self._forward),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self._tensors, str(path), metadata=metadata)
        return layer_captures.canonicalize_header(path)

    def stages_in_model_order(self) -> list[str]:
        """Embeddings first, final norm last, layers in index order (hook registration order already is that)."""
        return list(self.stages)
