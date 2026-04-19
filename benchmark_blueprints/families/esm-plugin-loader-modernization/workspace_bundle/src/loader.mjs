export async function loadPluginModule(name) {
  const mod = await import(`../plugins/${name}.js`);
  return mod.default ?? mod;
}
