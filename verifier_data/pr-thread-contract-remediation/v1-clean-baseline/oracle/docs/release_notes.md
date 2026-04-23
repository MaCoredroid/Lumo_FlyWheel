# Release Notes

## Queue summary API

- unowned summary buckets now omit `owner` instead of serializing `null`
- request-side owner filter semantics remain unchanged
- `include_unowned=true` appends the unowned bucket after the existing owner buckets without re-sorting them
