package dev.langchain4j.model.embedding.internal;

import java.io.IOException;

public class GpuUtils {

    public static boolean hasGpu() {
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.command("nvidia-smi");

        try {
            Process process = processBuilder.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }
}
