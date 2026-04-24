## RT-311

Fixed the unresolved mobile reopen on `/pull/241/review-thread/hotfix` at `412x915`.

The unresolved card no longer clips the title under the trailing action cluster: the meta row now wraps on mobile, the title can break onto multiple lines, and the actions move onto their own wrapped line instead of sitting flush against the text.

I also scoped the accessibility update to the actionable unresolved control only by naming `reply-thread-menu` as `Reply thread actions`, without reintroducing the rolled-back broad label applied to other menu buttons in INC-4481.
