export function buildReleaseGateRequest(formState: { releaseId: string; approvalState: string }) {
  return {
    path: `/api/releases/${formState.releaseId}/gate`,
    method: "POST",
    body: {
      release_id: formState.releaseId,
      approval_state: formState.approvalState || "manual_review",
    },
  };
}

export function summarizeGateEcho(response: { approval_state: string }) {
  return `Server echoed ${response.approval_state || "manual_review"}`;
}
