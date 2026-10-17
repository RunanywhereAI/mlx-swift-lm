// Copyright © 2026 RunAnywhere AI.

import Foundation

/// How the full-attention layers of a hybrid backbone mask their keys.
///
/// Generation is always `.causal`. Classification checkpoints trained with
/// every token visible to every other in the full-attention layers (the
/// pplx deciders, for one) need `.bidirectional`; their recurrent layers stay
/// causal by construction either way.
public enum FullAttentionMode: String, Sendable, Codable {
    case causal
    case bidirectional
}
