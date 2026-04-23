
export async function maybeLoadPlugin(name) {
  const mod = await import(new URL(`../plugins/${name}.mjs`, import.meta.url));
  return mod.default ?? mod;
}
