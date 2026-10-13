import { promises as fs } from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";

const wasmSource = path.resolve("node_modules/@mediapipe/tasks-vision/wasm");
const wasmRoute = "/mediapipe-wasm/";

function mediaPipeWasmAssets() {
  let outputDirectory;
  return {
    name: "mediapipe-wasm-assets",
    configResolved(config) {
      outputDirectory = path.resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        if (!request.url?.startsWith(wasmRoute)) return next();
        const filename = path.basename(new URL(request.url, "http://localhost").pathname);
        if (!filename.startsWith("vision_wasm_")) return next();
        try {
          const file = await fs.readFile(path.join(wasmSource, filename));
          response.setHeader("Content-Type", filename.endsWith(".wasm") ? "application/wasm" : "text/javascript; charset=utf-8");
          response.end(file);
        } catch {
          next();
        }
      });
    },
    async writeBundle() {
      await fs.cp(wasmSource, path.join(outputDirectory, "mediapipe-wasm"), { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [mediaPipeWasmAssets()],
});
