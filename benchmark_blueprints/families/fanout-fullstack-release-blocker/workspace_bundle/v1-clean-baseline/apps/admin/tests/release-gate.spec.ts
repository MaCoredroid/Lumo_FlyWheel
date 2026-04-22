import { buildReleaseGateRequest } from "../src/lib/api";

test("release gate request posts approval_state", () => {
  const request = buildReleaseGateRequest({
    releaseId: "rel-ship-0422",
    approvalState: "human_review_required",
  });
  expect(request.body.approval_state).toBe("human_review_required");
});
