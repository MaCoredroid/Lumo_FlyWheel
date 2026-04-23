
export function isPlugin(value) {
  return Boolean(value)
    && typeof value === "object"
    && typeof value.name === "string"
    && typeof value.run === "function";
}

export function assertPluginContract(name, value) {
  if (!isPlugin(value)) {
    throw new Error(`Invalid plugin module: ${name}`);
  }
  return value;
}
