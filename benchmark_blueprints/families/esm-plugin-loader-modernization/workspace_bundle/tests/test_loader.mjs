import test from "node:test";
import assert from "node:assert/strict";
import { loadPluginModule } from "../src/loader.mjs";

test("loads default export plugin", async () => {
  const plugin = await loadPluginModule("good-default");
  assert.equal(plugin.run("report"), "default:report");
});

test("loads named export plugin", async () => {
  const plugin = await loadPluginModule("good-named");
  assert.equal(plugin.run("report"), "named:report");
});
