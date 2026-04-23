
import { readFileSync } from "node:fs";

const loader = readFileSync(new URL("../src/loader.mjs", import.meta.url), "utf8");
const contracts = readFileSync(new URL("../src/contracts.mjs", import.meta.url), "utf8");
const index = readFileSync(new URL("../src/index.mjs", import.meta.url), "utf8");

const checks = [
  loader.includes("resolvePluginUrl"),
  loader.includes("../plugins/${name}.mjs"),
  loader.includes("assertPluginContract"),
  contracts.includes("assertPluginContract"),
  index.includes("dist/src/index.mjs")
];

if (checks.some((value) => !value)) {
  process.exit(1);
}
