package dev.langchain4j.model.embedding.bge.small.zh.v15;

import dev.langchain4j.model.embedding.AbstractInProcessEmbeddingModel;
import dev.langchain4j.model.embedding.OnnxBertBiEncoder;
import dev.langchain4j.model.embedding.PoolingMode;

/**
 * Quantized BAAI bge-small-zh-v1.5 embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Embedding dimensions: 512
 * <p>
 * It is recommended to add "为这个句子生成表示以用于检索相关文章：" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-zh-v1.5">here</a>
 */
public class BgeSmallZhV15QuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private final OnnxBertBiEncoder model;

    public BgeSmallZhV15QuantizedEmbeddingModel(boolean useCuda, int... cudaIds) {
        model = loadFromJar(
                "bge-small-zh-v1.5-q.onnx",
                "bge-small-zh-v1.5-tokenizer.json",
                PoolingMode.CLS,
                useCuda,
                cudaIds
        );
    }

    public BgeSmallZhV15QuantizedEmbeddingModel() {
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
