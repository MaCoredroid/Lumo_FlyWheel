export const defaultGateState = {
  releaseId: "rel-ship-0422",
  approvalState: "human_review_required",
  operatorLabel: "HumanReviewRequired",
};

export function buildSeededGateForm() {
  return { ...defaultGateState };
}
