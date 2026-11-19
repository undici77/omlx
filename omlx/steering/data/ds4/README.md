# DS4 direction-steering prompt sets

Imported from [antirez/ds4](https://github.com/antirez/ds4) at commit
`b3f50180`, `dir-steering/examples/`, MIT license (see `LICENSE`).

| File | Contents | Used for |
|---|---|---|
| `succinct.txt` | 100 terse-style prompts | the "good" side of a verbosity pack |
| `verbose.txt` | 100 matching verbose prompts (paired by topic) | the "bad" side |
| `eval_prompts.txt` | 5 held-out prompts | sweep / effect measurement (E1) |

Build a pack from them:

```bash
./steering_build.sh --good-file omlx/steering/data/ds4/succinct.txt \
                    --bad-file  omlx/steering/data/ds4/verbose.txt \
                    --name terse
```

`steering_build.sh` also picks these up as its defaults when present.
Vectors built from these prompts are tied to the model and hook point they
were captured on (plan C3) — rebuild per model.
