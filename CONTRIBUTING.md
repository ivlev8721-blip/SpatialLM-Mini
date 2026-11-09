# Contributing / Участие в проекте

Issues and pull requests are welcome in English or Russian. The most valuable contributions help distinguish a useful idea from an artifact of one implementation or dataset.

## Useful research tasks

1. Compare the same local + dyadic graph with learned 3D scores, ordinary dot-product scores, and fixed or uniform weights. Keep parameter count, data, training budget, and evaluation positions controlled.
2. Add a local-attention baseline with comparable optimized kernels.
3. Test retrieval of a fact or key/value pair at controlled distances. Report performance by distance and context length, with an untrained baseline.
4. Repeat on additional independent texts, longer training budgets, and more seeds.
5. Profile geometric gather operations and implement incremental generation or a KV cache.

These are open tasks, not features already implemented or guaranteed improvements.

## Before proposing a change

- Describe the problem and expected behavior in an issue or PR.
- Run `python -m unittest discover -s tests -v` from the repository root. Include the result and note skipped tests.
- For model changes, check causality, finite gradients, and save/load compatibility. Add tests where they verify a meaningful behavior.
- For benchmark claims, include the command, commit, dataset hashes, seeds, device, PyTorch/CUDA versions, precision, attention backend, parameter counts, timing scope, and raw results.
- Keep each new experiment in a fresh `runs/` directory. Do not overwrite recorded results. Publish concise measurement JSON under `results/` when appropriate; keep environments and model weights out of Git.
- Avoid tuning on test data. Explain all differences between models, including FFN width and kernels.

Original project code and documentation are MIT-licensed. Preserve existing copyright and third-party notices. Contributions are submitted under the repository's license.

## По-русски

Можно создавать issue и pull request на русском. Наиболее полезны проверки роли геометрии при неизменном графе, сравнение с локальным вниманием, тесты извлечения далёких фактов и независимые повторения. Для результатов приложите команды, сведения об окружении, данные и исходные измерения. Не перезаписывайте существующие опыты и не подбирайте параметры по test. Соблюдайте MIT-лицензию и сохраняйте авторские уведомления.
