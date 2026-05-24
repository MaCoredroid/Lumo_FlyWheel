export const defaultGateState = {
  releaseId: "rel-ship-0422",
  approvalState: "manual_review",
  operatorLabel: "Manual review",
};

export function buildSeededGateForm() {
  return { ...defaultGateState };
}
