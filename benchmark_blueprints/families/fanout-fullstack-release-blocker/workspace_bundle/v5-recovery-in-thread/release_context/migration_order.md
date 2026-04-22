# Release context

The release fanout reads the intercepted request payload before the server echo
appears in the operator console. If the runbook verifies echo first, operators
can falsely approve a stale UI submission.
