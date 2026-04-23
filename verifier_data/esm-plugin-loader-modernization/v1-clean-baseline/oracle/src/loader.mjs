
import { assertPluginContract } from "./contracts.mjs";

export function listPluginNames() {
  return ["good-default", "good-named", "good-helper", "bad-wrong-shape"];
}

export function resolvePluginUrl(name) {
  return new URL(`../plugins/${name}.mjs`, import.meta.url);
}

export async function loadPluginModule(name) {
  const mod = await import(resolvePluginUrl(name).href);
  if ("default" in mod && mod.default !== undefined) {
    return assertPluginContract(name, mod.default);
  }
  if ("plugin" in mod) {
    return assertPluginContract(name, mod.plugin);
  }
  throw new Error(`Invalid plugin module: ${name}`);
}
