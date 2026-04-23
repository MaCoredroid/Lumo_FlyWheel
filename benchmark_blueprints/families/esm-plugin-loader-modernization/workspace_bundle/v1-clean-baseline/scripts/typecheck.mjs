
import { readFileSync } from "node:fs";

const loader = readFileSync(new URL("../src/loader.mjs", import.meta.url), "utf8");
const contracts = readFileSync(new URL("../src/contracts.mjs", import.meta.url), "utf8");

const checks = [
  loader.includes("resolvePluginUrl"),
  loader.includes("assertPluginContract"),
  contracts.includes("assertPluginContract")
];

if (checks.some((value) => !value)) {
  process.exit(1);
}
