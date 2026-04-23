
import { listPluginNames, loadPluginModule } from "./loader.mjs";

const arg = process.argv[2];

if (!arg || arg === "--help") {
  console.log([
    "Usage: node dist/src/index.mjs <plugin-name>",
    "Commands:",
    "  --list   Print discoverable plugin names",
    "Notes:",
    "  - Supports default-export and named-export plugins",
    "  - Rejects malformed plugin modules with Invalid plugin module"
  ].join("\n"));
  process.exit(0);
}

if (arg === "--list") {
  console.log(listPluginNames().join("\n"));
  process.exit(0);
}

const plugin = await loadPluginModule(arg);
console.log(plugin.run("report"));
