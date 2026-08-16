# RunAnywhere mlx-swift-lm fork

This fork tracks canonical `ml-explore/mlx-swift-lm`. It exists for two narrow
reasons:

1. all transitive packages must resolve RunAnywhere's `mlx-swift` runtime so
   PrismML Bonsai 1-bit kernels are never silently replaced by upstream; and
2. DeepGrove Maple Preview needs a native Swift model implementation.

The Maple graph is a Swift port of the MIT-licensed reference in
`deepgrove-ai/mlx-lm-deepgrove`; its fused Python/Metal decode kernels are not
copied. The portable graph preserves the trained forward pass and checkpoint
layout while keeping the first release easy to audit against stock MLX ops.

Release tag `3.31.5` is RunAnywhere-local bookkeeping based on canonical main
after upstream `3.31.4`. Maple starts with the portable stock-MLX graph; fused
decode kernels may be added only after bitwise/coherence tests prove parity.
