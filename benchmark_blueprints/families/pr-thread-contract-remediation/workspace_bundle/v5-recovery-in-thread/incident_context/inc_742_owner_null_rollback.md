# INC-742 owner-null rollback

The prior hotfix serialized `owner: null` for unowned buckets. Mobile
parity checks failed, so the patch was rolled back. The recovery path is
to omit the key and document the contract clearly.
