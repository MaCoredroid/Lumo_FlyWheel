# Abandoned patch

A previous attempt only blocked the literal string `..` and used a
string-prefix ancestry check. AppSec rejected it because encoded
separators, mixed slashes, and symlink escapes still crossed the
tenant root.
