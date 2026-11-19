# DS4 direction-steering prompt sets

Imported from [antirez/ds4](https://github.com/antirez/ds4) at commit
`b3f50180`, `dir-steering/examples/`, MIT license (see `LICENSE`).

| File | Contents | Used for |
|---|---|---|
| `succinct.txt` | 100 terse-style prompts | the "good" side of a verbosity pack |
| `verbose.txt` | 100 matching verbose prompts (paired by topic) | the "bad" side |
| `eval_prompts.txt` | 5 held-out prompts | sweep / effect measurement |

Build a pack from them:

```bash
omlx steer build MODEL --preset verbosity --name terse
```

`--preset verbosity` uses `succinct.txt` / `verbose.txt`; `omlx steer sweep` uses
`eval_prompts.txt` when `--prompts` is omitted.
Vectors built from these prompts are tied to the model and hook point they
were captured on — rebuild per model.
