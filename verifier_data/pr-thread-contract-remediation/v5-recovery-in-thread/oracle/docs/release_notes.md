# Release Notes

## Queue summary API

- unowned summary buckets now omit `owner` instead of serializing `null`
- request-side owner filter semantics remain unchanged
- `include_unowned=true` appends the unowned bucket after the existing owner buckets without re-sorting them
- mobile compatibility depends on keeping the omit-not-null response contract
- this follows the INC-742 rollback of the earlier `owner: null` hotfix
