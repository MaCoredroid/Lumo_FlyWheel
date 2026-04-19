export function SettingsPanel(optionalValue?: string) {
  const optionalValueSummary = optionalValue
    ? `<p>Optional connector value: ${optionalValue}</p>`
    : "<p>Optional connector value is unset. Connector fallback remains available.</p>";

  return `<section><h1>Settings</h1>${optionalValueSummary}<label><input type="checkbox" name="connector_fallback" />Connector fallback</label></section>`;
}
