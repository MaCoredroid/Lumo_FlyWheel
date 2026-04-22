export const TABLE_COLUMNS = ['workspace', 'risk_level', 'approval_state'];
export const APPROVAL_FALLBACK_BADGE = 'manual_review';

export function WorkspaceTable(props: { title: string; filterMode: string }) {
  return `${props.title}: ${TABLE_COLUMNS.join(',')} filtered-by=${props.filterMode}`;
}

export function renderApprovalStateBadge(row: { approval_state?: string }) {
  return row.approval_state || APPROVAL_FALLBACK_BADGE;
}

export const APPROVAL_STATE_COLUMN_LABEL = 'Approval state';
