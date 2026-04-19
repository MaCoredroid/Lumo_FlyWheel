import { readFileSync } from "node:fs";

const text = readFileSync(new URL("../src/loader.mjs", import.meta.url), "utf8");
if (!text.includes("loadPluginModule")) {
  process.exit(1);
}
