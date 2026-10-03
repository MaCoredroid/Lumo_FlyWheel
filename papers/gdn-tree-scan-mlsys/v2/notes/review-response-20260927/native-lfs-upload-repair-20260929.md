# Native A backup transfer repair

The first upload attempt failed and discarded per-object status; its exact cause cannot be recovered. The second attempt was stopped only after a later scoped upload established a successful PUT followed by verify HTTP403. No experiment ran during the repair and no Git branch was advanced.

Fresh local verification with explicit LFS headers and User-Agent accepted that last chunk (HTTP200). A fresh verify-only pass then registered the first20 uploaded chunks and confirmed the last chunk, while13 other objects returned404. These statuses prove stored versus missing objects; they do not by themselves isolate whether credentials, request headers or another server rule caused the earlier403.

The successor uploader adds explicit content negotiation and User-Agent, retains upload versus verify stage and numeric HTTPstatus without logging capabilities, and obtains fresh actions in groups of four. Existing registered objects are skipped. All artifacts remain unchanged; failures and the exact owned-uploader stop receipt are retained. Git publication still requires all34 native chunks, source archives and remote pointer/object verification.
