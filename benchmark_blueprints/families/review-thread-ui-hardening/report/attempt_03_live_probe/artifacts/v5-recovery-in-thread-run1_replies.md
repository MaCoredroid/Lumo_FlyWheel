RT-311

Fixed the recovery reopen on `/pull/241/review-thread/hotfix` at `412x915`. The unresolved card now gives the `reply-thread-menu` button a specific accessible name, and the mobile metadata row wraps the long title instead of clipping it under the trailing action cluster. I kept the change scoped to the unresolved recovery path and did not repeat the rolled-back broad label or overflow-hidden approach from `INC-4481`.
