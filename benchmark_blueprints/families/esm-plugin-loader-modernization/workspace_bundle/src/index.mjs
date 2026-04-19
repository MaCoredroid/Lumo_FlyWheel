import { loadPluginModule } from "./loader.mjs";

const plugin = await loadPluginModule(process.argv[2] || "good-default");
console.log(plugin.run("report"));
