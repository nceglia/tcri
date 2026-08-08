# Utilities (`tcri.ut`)

Session persistence — save and restore a fitted model together with its `AnnData`.
Exposed as ``tcri.ut``.

```{eval-rst}
.. autofunction:: tcri.utils._utils.save_tcri_session
.. autofunction:: tcri.utils._utils.load_tcri_session
```

```{note}
Only `save_tcri_session` and `load_tcri_session` are part of the public API
contract. Other names reachable under `tcri.ut` are internal helpers and may
change without notice.
```
