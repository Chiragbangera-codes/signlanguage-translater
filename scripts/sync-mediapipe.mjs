/**
 * Copy MediaPipe Hands runtime assets from node_modules into frontend/public.
 *
 * The app loads them same-origin rather than from a CDN: ad blockers and
 * network filters commonly block cdn.jsdelivr.net, which MediaPipe reports as
 * "Failed to fetch" on every frame with no recovery path. Serving our own copy
 * also keeps recognition working offline.
 *
 * Re-run after changing the @mediapipe/hands version:  npm run sync:mediapipe
 */
import { cp, mkdir, readdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "frontend", "node_modules", "@mediapipe", "hands");
const dest = join(here, "..", "frontend", "public", "mediapipe");

// Runtime assets only — the JS wrapper is imported from node_modules directly.
const KEEP = /\.(wasm|data|binarypb|tflite)$|_wasm_bin\.js$|_assets_loader\.js$/;

const files = (await readdir(src)).filter((f) => KEEP.test(f));
if (files.length === 0) {
  console.error(`No MediaPipe assets found in ${src}. Run npm install first.`);
  process.exit(1);
}

await mkdir(dest, { recursive: true });
let bytes = 0;
for (const file of files) {
  await cp(join(src, file), join(dest, file));
  bytes += (await stat(join(dest, file))).size;
}

console.log(`Copied ${files.length} files (${(bytes / 1e6).toFixed(1)} MB) to public/mediapipe`);
