
import { listPluginNames, loadPluginModule } from "./loader.mjs";

const arg = process.argv[2];

if (!arg || arg === "--help") {
  console.log([
    "Usage: node src/index.mjs <plugin-name>",
    "Commands:",
    "  --list   Print discoverable plugin names"
  ].join("\n"));
  process.exit(0);
}

if (arg === "--list") {
  console.log(listPluginNames().join("\n"));
  process.exit(0);
}

const plugin = await loadPluginModule(arg);
console.log(plugin.run("report"));
