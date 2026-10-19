# Starlings.Roost.ModelCompiler

A Roslyn incremental source generator (netstandard2.0, R-020). It reads a model definition written as attributes on a
`static partial class`, builds the op graph (contract C7, `Starlings.Roost.Graph`) and lowers it to
`Forward<TOps>()` over `IRoostOps` (contract C3). The op vocabulary is read from the `IRoostOps` symbol in the
compilation, so the generator lists no op names. It ships two ways: as an analyzer package for people who define new
models, and as a project reference for the `Starlings.Roost.Models.*` packages, whose assemblies carry the generated code.

```xml
<!-- analyzer package -->
<PackageReference Include="Starlings.Roost.ModelCompiler" Version="..." PrivateAssets="all" />
<!-- inside the repository -->
<ProjectReference Include="../Starlings.Roost.ModelCompiler/Starlings.Roost.ModelCompiler.csproj" OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
<ProjectReference Include="../Starlings.Roost.Graph/Starlings.Roost.Graph.csproj" OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
```

`Starlings.Roost.Graph.dll` is bundled next to the generator (`analyzers/dotnet/cs/` in the package), because an analyzer
cannot take package dependencies (S4 finding 3). The consuming project references `Starlings.Roost.Ops` and
`Starlings.Roost.Abstractions` normally; it references `Starlings.Roost.Graph` too if it wants `BuildGraph()`.

## The definition

The attributes are emitted into the consuming compilation (post-initialisation output) in the namespace
`Starlings.Roost.ModelCompiler`, as internal types.

```csharp
[RoostModel("tiny-dense", Dtype = "F32")]                       // activation dtype, an element type name
[Dim("hidden", 16)] [Dim("inter2", "2*inter")]                  // integer constants; expressions over earlier dims
[Weight("l{i}.norm", "hidden", Source = "model.layers.{i}.input_layernorm.weight", Layers = "0:2")]
[StateGroup("attn", "Full", Layers = "0:24%4=3", KvHeads = "kv_heads", HeadSize = "head_dim", BlockSize = "16")]
[Node("l{i}.norm", "RmsNorm", "Input=h{i}; Weight=l{i}.norm; Output=>l{i}.x[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:2")]
[Output("logits")]
public static partial class TinyDense { }
```

* **Layers** is a comma-separated list of `a:b` ranges (`b` excluded) or single indices, each with an optional
  `%m=r` or `%m!=r` filter. `{i}`, `{i+1}` and `{i-1}` in names, shapes, sources and arguments become the layer number.
  A run of consecutive attributes that all have `Layers` is a block, emitted layer by layer in declaration order, so a
  hybrid model's three recurrent layers and one attention layer each keep their nodes together. Keep a model's
  attributes in one file: the order of attributes across partial declarations is the compiler's, not yours.
* **Node arguments** are `FIELD=binding` pairs separated by `;`, named after the fields of the op's argument struct.
  A buffer field is bound to a value (`Input=h0`), written as a new value (`Output=>x[tokens,hidden]:BF16`; the
  shape's leading dimension may be `tokens` or `selected`, and the dtype defaults to the model's) or overwritten in
  place (`Q=q=>q_rot`, which makes `q_rot` a new value in the graph that shares `q`'s storage; `q` must not be read
  afterwards). A value is a node output, a weight slot (`w:name`), a state slot (`s:group.role` for the current layer,
  `s:group.l3.role` for a named one: roles `k`, `v`, `state`, `conv`) or a plan input (`p:Ids`). Scalar fields take an
  integer expression (over dims and the step symbols `tokens`, `selected`, `seqs`, `kv_blocks`), a float, `true` or
  `false`, or an enum member name.
* **Filled for you**: a buffer field a node leaves unbound and whose name is `Ids`, `Positions`, `SlotMapping`, `QIndptr`,
  `KvIndptr`, `KvIndices`, `KvLastPageLength`, `Rows`, `StateSlots` or `CuSeqlens` reads the step's plan input of that name;
  a `DType` field gets the model dtype; `KvDType` gets the dtype of the state slot bound in the node. Other unbound
  fields stay zero (optional buffers, `Bias`, `D`).

## What is generated

For `class M` in namespace `N`, a partial `M` with:

| Member | Content |
| --- | --- |
| `WeightSlots` | The loader's slot order: name, checkpoint source, dtype, elements, shape, layer. `StepContext.Weights[i]` is slot `i` |
| `StateSlots`, `CreateStateGroups()` | One slot per layer and role in each group (`attn.l3.k`, `linear.l0.conv`), and the C4 `StateGroupDescriptor`s (kind, layer KV shapes, block size, dtype, bytes per block, window, recurrent state shape, snapshot slots). A recurrent group's convolution state is part of its slot (`BytesPerBlock` sums both). Recurrent state defaults to `F32`, convolution state to the model dtype |
| `PlanInputs`, `ForwardLayout` | The step inputs the ops read (derived from the C3 argument structs they use), their dtype and count, and `Bind` to slice one region. `ForwardLayout` sizes the activation arena and the plan region from the largest step; token capacity is rounded up to 64 so every buffer is at least 16-byte aligned |
| `StepContext` | A ref struct over caller-owned memory: arena, weight spans, state spans, plan inputs and the step's counts |
| `Forward<TOps>(in StepContext)` | The identity lowering of the unfused graph, one block of struct initialisation and a static call per node. No allocation, no reflection, no virtual dispatch. A fused op with a known unfused form (`AddRmsNorm`) is emitted as `if (TOps.SupportsAddRmsNorm) fused else Residual + RmsNorm` |
| `Output_<name>(in StepContext)` | The span of a model output after the forward |
| `CheckBackend<TOps>()` | Throws at load, naming the reserved ops (`GatedDeltaRule`, `CausalConv1d`, ...) the backend does not support; the abstract ops are checked by the compiler (CS0535) |
| `BuildGraph()` | The C7 graph (only when `Starlings.Roost.Graph` is referenced) |

Activations share the arena by live range, and an in-place output shares its input's place. Token-major values are laid
out per token of capacity, so offsets scale with `ForwardLayout.CapacityTokens`.

### The graph protocol

Everything the lowering needs is in the graph, so the output of a pass can be lowered or evaluated the same way.
`OpNode.Attributes` holds `in` and `out` (comma-separated argument-field names, one per `Inputs` and `Outputs` position;
a field in both is in place), `arg.FIELD` (a scalar argument, with dims folded into integer expressions), `node` (the
definition's node name) and `layer`. Value ids index `OpGraph.Values`: weight slots first, then state slots, then plan
inputs and activations in order of first use. Plan inputs are `ValueKind.Input` named `plan.<Field>`; state slots are
`ValueKind.State` with a leading `blocks` or `slots` dimension.

## The graph checker

`Checking.GraphChecker` implements `IGraphChecker`: it evaluates the unfused reference graph and a pass's output with
the same `IGraphEvaluator` and compares, for every candidate node, each value it produces with the reference value of
the same id, with the tolerance the `IToleranceSource` gives for the op and dtype. A mismatch names the candidate node and
the reference nodes it replaced (`Origin`); a reference model output the candidate does not produce is a mismatch too.
`CheckPass` runs a pass first. The evaluator on `CpuOps` is stream D's; the tests use a stub backend and a reflection
evaluator over the node attributes above.

## Diagnostics

| Id | Meaning |
| --- | --- |
| ROOST001 | Op is not a member of `IRoostOps` (names the node and lists the known ops) |
| ROOST002 | A name is not an earlier node output, weight slot, state slot or plan input |
| ROOST003 | A field is not a member of the op's argument struct |
| ROOST004 | A binding does not fit its field (bad scalar, enum member, in-place write of a weight) |
| ROOST005 | `IRoostOps` is not referenced |
| ROOST006 | A shape, dim or dtype is invalid, or an output has no shape |
| ROOST007 | A name is declared twice |
| ROOST008 | The definition is invalid (no `[Output]`, bad model dtype) |
| ROOST009 | A value is read after an op overwrote it in place |
| ROOST010 | A layer set or placeholder is invalid |
| ROOST011 | A state group is invalid |
| ROOST012 | A fused op has no known unfused form |
| ROOST013 | A contract type the generated code needs is not referenced |
| ROOST014 | (warning) A weight slot is never read |

A node whose op or arguments fail is reported once; nodes that read its outputs are not reported again.

## Known limits

* The forward is unrolled: 24 layers of a Qwen3.5-shaped model are about 6 200 generated lines in one method. Loop
  emission over a per-layer weight table is not built (S4 "first thing to try").
* There is no view binding: a node binds a whole value to a buffer field, so it cannot yet read a slice of a value.
  Since contract version 2, `GatedDeltaRuleArgs` takes row-strided q, k and v (`QStride`, `KStride`, `VStride`, bindable
  as scalars; 0 means contiguous), so Qwen3.5's fused q/k/v conv output can be read in place once a definition can bind
  a column slice (for example `Q=l{i}.c[0:key_dim]`); until then a definition needs separate projections where a
  checkpoint fuses them.
* Only `AddRmsNorm` has an unfused form in the lowering; other fused ops come from passes (stream L).
* Plan-input auto-fill goes by field name, so an unbound `Ids` span is the token-id input for every op that has one. For
  `Argmax` and `Sample`, `Ids` is an output; bind it explicitly if a definition ever includes them (the forward ends at
  the logits and the worker launches the sampler).
* The generated class has no `unsafe` code and needs no `AllowUnsafeBlocks` in the consumer.
* The generator adds source locations to its pipeline model, so inserting a line above the attributes re-emits.
