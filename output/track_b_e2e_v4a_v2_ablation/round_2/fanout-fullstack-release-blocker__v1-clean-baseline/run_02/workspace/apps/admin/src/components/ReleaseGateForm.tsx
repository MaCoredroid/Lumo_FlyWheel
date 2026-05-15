export const defaultGateState = {
  releaseId: "rel-ship-0422",
  approvalState: "human_review_required",
  operatorLabel: "Human review required",
};

export function buildSeededGateForm() {
  return { ...defaultGateState };
}
