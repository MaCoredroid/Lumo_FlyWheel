
export function listPluginNames() {
  return ["good-default", "good-named", "good-helper", "bad-wrong-shape"];
}

export async function loadPluginModule(name) {
  const mod = await import(new URL(`../plugins/${name}.js`, import.meta.url));
  return mod.default ?? mod;
}
