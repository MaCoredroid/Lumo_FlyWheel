# FR13_RESHAPE_WIDE sweep sequence (sourced by fr13_b4_campaign_driver.sh;
# run_native/run_variant are in scope from the driver).
#
# Goal (user 2026-06-17): measure the wider-not-deeper axis at B=1 temp-0.6 on
# the SWE-Verified deploy, depth-matched:
#   cat55222 = [5,5,2,2,2]  16 nodes, depth-5  -> vs native E5 (banked)
#   cat555   = [5,5,5]      15 nodes, depth-3  -> vs native E3 (captured here)
# Same 4 tasks as the existing E5/E3 bars (subset_b4_four.json) for an
# apples-to-apples token-weighted decode-throughput compare + lossless.
#
# Order is RISK-FRONT-LOADED so a build problem surfaces at warmup (~15min, the
# EXPECT_RATIO draft-shape assert) instead of after a 2hr eval:
#   1. cat555   (15 nodes) — first exercise of the general width-N drafter
#   2. cat55222 (16 nodes) — the N_PAD=16 h_cache boundary, second wide arm
#   3. nativeE3 (MTP-3)    — the depth-3 bar; native, lowest risk, last
run_variant cat555_${TAG}      cat555    15 1
run_variant cat55222_${TAG}    cat55222  16 1
run_native  nativeE3_${TAG}    3 3 1
