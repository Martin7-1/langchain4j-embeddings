package dev.langchain4j.model.embedding;

/**
 * Quantized BAAI bge-small-zh embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Embedding dimensions: 512
 * <p>
 * It is recommended to add "为这个句子生成表示以用于检索相关文章：" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-zh">here</a>
 */
public class BgeSmallZhQuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private final OnnxBertBiEncoder model;

    public BgeSmallZhQuantizedEmbeddingModel(boolean useCuda, int... cudaIds) {
        model = loadFromJar(
                "bge-small-zh-q.onnx",
                "tokenizer.json",
                PoolingMode.CLS,
                useCuda,
                cudaIds
        );
    }

    public BgeSmallZhQuantizedEmbeddingModel() {
        this(false);
    }

    @Override
    protected OnnxBertBiEncoder model() {
        return model;
    }

    @Override
    protected Integer knownDimension() {
        return 512;
    }
}
