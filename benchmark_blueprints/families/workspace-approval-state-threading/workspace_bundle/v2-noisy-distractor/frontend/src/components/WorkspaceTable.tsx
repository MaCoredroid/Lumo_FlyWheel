export const TABLE_COLUMNS = ['workspace', 'risk_level'];

export function WorkspaceTable(props: { title: string }) {
  return `${props.title}: ${TABLE_COLUMNS.join(',')}`;
}

export function renderRiskBadge(row: { risk_level: string }) {
  return row.risk_level;
}
