## Retro Note

The temporary compatibility shim kept one older rehearsal stable, but the note
was written before the service adopted the current worker retry path. Do not
assume the shim is still the safest answer without checking transaction
behavior in the current code.
