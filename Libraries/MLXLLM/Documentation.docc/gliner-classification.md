# GLiNER label classification

`GLiNERClassifier` loads `souuzaa/GLiNER2.5-Decide-4bit` locally and scores supplied labels. It implements the DeBERTa encoder and classification head. Span, entity, relation and structure extraction are outside this API.

```swift
let model = try await GLiNERClassifier.load(from: directory, using: tokenizerLoader)
let results = try model.classify(
    text: "I was charged twice.",
    tasks: [
        GLiNERClassificationTask(
            name: "intent",
            labels: [.init("refund"), .init("cancel"), .init("other")])
    ])
```

Each result preserves label order and exposes raw logits, all probabilities and selected label names. Automatic activation uses softmax for single-label tasks and sigmoid for multi-label tasks. Explicit activation overrides that choice. Temperature scales logits before activation. Single-label tasks select the first maximum; multi-label tasks select probabilities at or above the threshold, falling back to the first maximum if none qualify.

Task names and label names must be nonempty and unique within their respective sets. Temperatures must be finite and positive; thresholds must be in [0, 1]. Reserved schema markers cannot occur in user fields. Requests exceeding `maxTokens` fail instead of truncating. `maxWords` optionally truncates text words explicitly, matching the reference preprocessing. Empty text becomes a period. Text words are lowercased; labels and task prompts retain their case.

The default load retains checkpoint BF16 float parameters and 4-bit affine encoder weights. `floatingPointType: .float32` casts float parameters, including quantization scales, while preserving packed 4-bit weights. Relative embeddings and the classification head stay unquantized. Known extraction-only tensors are checked for their exact shapes and floating dtype before discarding; unknown or missing model tensors fail strict loading. Use one classifier instance sequentially.

The caller supplies an `MLXLMCommon.TokenizerLoader` that supports the checkpoint's DeBERTa Unigram tokenizer. swift-transformers 1.3.4 does not register `DebertaV2Tokenizer`, and its Unigram model ignores added token IDs. `GLiNERTokenizerConfiguration.load(from:)` adapts the JSON in memory: it selects the existing Unigram implementation and extends its vocabulary with the checkpoint's exact added token IDs. It preserves the normalizer, added token matching and preprocessor. Decode the returned `tokenizerConfig` and `tokenizerData` into the caller's tokenizer library configuration types. The real-model test loader demonstrates this adapter and verifies exact token IDs against Python. Cached files stay unchanged.

`extractor` is registered in the generative LLM registry with a `GLiNERError.classificationOnly` rejection directing callers to this typed loader. GLiNER has no vocabulary-generation head, so returning a generative `LanguageModel` would misrepresent its behavior.

Checkpoint/reference revision: `6023a32c2e696529decacfcaeb683b62bc535bdf`. Real tests require `GLINER_REAL_MODEL_TESTS=1`; `GLINER_REAL_MODEL_PATH` overrides the default shared cache directory. Once opted in, missing files fail. Both BF16 and float32 tests compare seven inputs, exact preprocessing IDs, selected labels and every label score against the checkpoint's `gliner2_mlx.py`.
