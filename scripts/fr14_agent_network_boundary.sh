#!/bin/bash
# FR14 agent network boundary — runs ON ALIENWARE (the agent host).
#
# Why: instance containers ran --network=host, giving the SWE agent full
# internet; Qwen3.8 used it (shell python one-liners) to fetch gold patches
# (see results/fr14_nvfp4_port_20260816/agent_network_egress_finding.md).
# qwen-code's shell-equivalence denylist is a heuristic and unwinnable.
#
# Fix: a dedicated b