
export function isPlugin(value) {
  return Boolean(value) && typeof value.run === "function";
}
