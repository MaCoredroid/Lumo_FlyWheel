# FR13_RESHAPE_WIDE cat55221 re-run sequence (sourced by fr13_b4_campaign_driver.sh).
# cat55222 [5,5,2,2,2]=16 nodes overflowed the verifier cap (n_pad=next_pow2(17)=32>16);
# cat55221 [5,5,2,2,1]=15 nodes is the FITTING depth-5 wide arm (n_pad=16). vs banked E5.
# Same 4 tasks (subset_b4_four.json), B=1 temp-0.6, same infra as cat555/E5/E3.
run_variant cat55221_b1 cat55221 15 1
