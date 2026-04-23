# Mobile client contract

iOS 7.4 treats an omitted owner key as unassigned, but `owner: null`
is parsed as an explicit cleared assignment. Preserve the response
contract and do not change request-side owner filter semantics.
