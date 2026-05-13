export const defaultGateState = {
  releaseId: "rel-ship-0422",
  approvalState: "human_review_required",
  operatorLabel: "Human_review_required",
};

export function buildSeededGateForm() {
  return { ...defaultGateState };
}
