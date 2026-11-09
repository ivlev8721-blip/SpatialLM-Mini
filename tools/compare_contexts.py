"""Validate and summarize complete context sweeps; no model dependencies."""
import argparse
import json
import math
from pathlib import Path
import statistics


ARCHS = ("spatial", "transformer", "hybrid")
NAMES = {"spatial": "Пространственная", "transformer": "Transformer", "hybrid": "Гибрид"}


def compare(root):
    root = Path(root)
    sources = {128: root / "context128_common/results.json",
               256: root / "context256_v1/results.json",
               512: root / "context512_v1/results.json"}
    experiments = {context: json.loads(path.read_text(encoding="utf-8")) for context, path in sources.items()}
    reference = experiments[128]["protocol"]
    summaries, paired = [], []
    for context, result in experiments.items():
        protocol, records = result["protocol"], result["records"]
        if len(records) != 18:
            raise ValueError(f"context {context}: expected 18 completed measurements")
        if protocol["data_sha256"] != reference["data_sha256"]:
            raise ValueError("corpora differ between contexts")
        if protocol["evaluation_prefix_sha256"] != reference["evaluation_prefix_sha256"]:
            raise ValueError("evaluation positions differ between contexts")
        args = protocol["arguments"]
        if args["batch_size"] * args["seq_len"] != 1024 or args["steps"] != 500 or args["seconds"] != 45:
            raise ValueError("incompatible training budget")
        for key in ("dim", "layers", "heads", "local_window", "lr", "weight_decay", "seeds"):
            if args[key] != reference["arguments"][key]:
                raise ValueError(f"training setting differs: {key}")
        if protocol["precision"] != reference["precision"]:
            raise ValueError("precision differs")
        for row in records:
            if row["model_config"]["max_seq_len"] != context:
                raise ValueError("checkpoint context does not match group")
            if any(row[part]["targets"] != 70144 for part in ("validation", "test")):
                raise ValueError("unexpected evaluation target count")
            if row["comparison"] == "equal_steps" and row["training_tokens"] != 512000:
                raise ValueError("different token budget at 500 steps")
            if row["comparison"] == "equal_steps" and row["steps"] != 500:
                raise ValueError("different update count")
            if row["training_tokens"] != row["steps"] * 1024:
                raise ValueError("training token and step counts disagree")
        for comparison in ("equal_steps", "equal_time"):
            for arch in ARCHS:
                selected = [r for r in records if r["architecture"] == arch and r["comparison"] == comparison]
                if sorted(r["seed"] for r in selected) != [42, 43, 44]:
                    raise ValueError("missing or duplicate seed")
                entry = dict(context=context, architecture=arch, comparison=comparison,
                             parameters=selected[0]["parameters"], batch_size=args["batch_size"])
                metrics = {
                    "test_loss": [r["test"]["loss"] for r in selected],
                    "validation_loss": [r["validation"]["loss"] for r in selected],
                    "steps": [r["steps"] for r in selected],
                    "training_seconds": [r["training_seconds"] for r in selected],
                    "tokens_per_second": [r["training_tokens_per_second"] for r in selected],
                    "peak_allocated_mib": [r["peak_allocated_mib"] for r in selected],
                }
                for metric, values in metrics.items():
                    entry[metric + "_mean"] = statistics.mean(values)
                    entry[metric + "_sd"] = statistics.stdev(values)
                summaries.append(entry)
            for arch in ("spatial", "hybrid"):
                deltas = []
                for seed in (42, 43, 44):
                    candidate = next(r for r in records if r["architecture"] == arch and r["seed"] == seed and r["comparison"] == comparison)
                    baseline = next(r for r in records if r["architecture"] == "transformer" and r["seed"] == seed and r["comparison"] == comparison)
                    deltas.append(candidate["test"]["loss"] - baseline["test"]["loss"])
                paired.append(dict(context=context, architecture=arch, comparison=comparison,
                                   deltas=deltas, mean_delta=statistics.mean(deltas),
                                   sd_delta=statistics.stdev(deltas), wins=sum(d < 0 for d in deltas),
                                   perplexity_change_percent=100 * math.expm1(statistics.mean(deltas))))
    return {"summary": summaries, "paired_deltas": paired,
            "evaluation_targets_per_partition": 70144, "experiments": experiments}


def report(result):
    rows, pairs = result["summary"], result["paired_deltas"]
    lines = ["# SpatialLM-Mini: контекст 128, 256 и 512 байт", "",
             "Локальная проверка 9 ноября 2026 года на RTX 3050 8 ГБ. На контекстах 256 и 512 проведено 18 новых обучений с нуля: три архитектуры × три seed × две длины. Для контекста 128 повторно оценены 18 сохранённых состояний предыдущего опыта; веса и время их обучения сохранены без изменений. В сумме сравниваются 54 состояния.", "",
             "**Это длина в UTF-8 байтах.** По отношению к первому опыту контекст увеличен вдвое и вчетверо; 512 байт не следует приравнивать к тысячам обычных LLM-токенов.", "",
             "## Качество при одинаковом времени обучения", "",
             "Каждой модели выделено 45 секунд накопленного времени шагов обучения. Прогрев, оценка, сохранение и генерация в бюджет не входят. Ошибка — negative log-likelihood на проверочном тексте: меньше — лучше. Среднее ± стандартное отклонение по seed 42, 43, 44; это не доверительный интервал.", "",
             "| Контекст, байт | Пространственная | Transformer | Гибрид |", "|---:|---:|---:|---:|"]
    for context in (128, 256, 512):
        cells = []
        for arch in ARCHS:
            row = next(r for r in rows if r["context"] == context and r["architecture"] == arch and r["comparison"] == "equal_time")
            cells.append(f"{row['test_loss_mean']:.4f} ± {row['test_loss_sd']:.4f}")
        lines.append(f"| {context} | " + " | ".join(cells) + " |")
    lines += ["", "## Парное сравнение с Transformer", "",
              "Отрицательная разница ошибки означает преимущество кандидата при одинаковом времени. Процент perplexity вычислен как `100*(exp(средняя разница loss)-1)`.", "",
              "| Контекст | Модель | Разница test loss | Выигрышей из 3 | Разница perplexity |", "|---:|---|---:|---:|---:|"]
    for pair in pairs:
        if pair["comparison"] == "equal_time":
            lines.append(f"| {pair['context']} | {NAMES[pair['architecture']]} | {pair['mean_delta']:+.4f} | {pair['wins']} | {pair['perplexity_change_percent']:+.2f}% |")
    lines += ["", "## Скорость и память", "",
              "Средние показатели по трём запускам при 45-секундном бюджете. Память — максимум выделенной PyTorch памяти, без памяти драйвера и рабочего стола.", "",
              "| Контекст | Модель | Шагов | Обучающих байт/с | Пик памяти, МиБ | Параметров |", "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["comparison"] == "equal_time":
            lines.append(f"| {row['context']} | {NAMES[row['architecture']]} | {row['steps_mean']:.0f} | {row['tokens_per_second_mean']:.0f} | {row['peak_allocated_mib_mean']:.1f} | {row['parameters']} |")
    lines += ["", "## Качество при одинаковом объёме обучения", "",
              "После 500 шагов каждая модель обработала 512000 целевых байт. Во всех длинах шаг содержит 1024 байта: batch 8×128, 4×256 либо 2×512. Learning rate и количество обновлений одинаковы.", "",
              "| Контекст, байт | Пространственная | Transformer | Гибрид |", "|---:|---:|---:|---:|"]
    for context in (128, 256, 512):
        cells = []
        for arch in ARCHS:
            row = next(r for r in rows if r["context"] == context and r["architecture"] == arch and r["comparison"] == "equal_steps")
            cells.append(f"{row['test_loss_mean']:.4f} ± {row['test_loss_sd']:.4f}")
        lines.append(f"| {context} | " + " | ".join(cells) + " |")
    lines += ["", "## Вывод по измеренным режимам", ""]
    for context in (128, 256, 512):
        selected = [r for r in rows if r["context"] == context and r["comparison"] == "equal_time"]
        best = min(selected, key=lambda r: r["test_loss_mean"])
        pair = next(p for p in pairs if p["context"] == context and p["architecture"] == "hybrid" and p["comparison"] == "equal_time")
        lines.append(f"- Контекст {context}: минимальная средняя ошибка у модели **{NAMES[best['architecture']]}**. Гибрид выигрывает у Transformer в {pair['wins']} из 3 парных запусков.")
    lines += ["", "Эти результаты относятся к конкретным реализациям и короткому бюджету. Более длинный контекст может менять соотношение моделей; сам факт его увеличения не гарантирует лучшего качества. Быстродействие следует оценивать отдельно от ошибки.", "",
              "## Что обеспечивает сопоставимость", "",
              "- Тот же сохранённый текст *Pride and Prejudice* Джейн Остин, [Project Gutenberg](https://www.gutenberg.org/ebooks/1342), те же train/validation/test и их SHA-256. Новых данных в обучение не добавлялось.",
              "- Во всех опытах оценены ровно одни и те же 70144 целевых байта validation и 70144 test, начиная с позиции 1. Выбранный размер кратен 128, 256 и 512. Остатки вне общего префикса исключены одинаково: 228 возможных целевых байт validation и 275 test. Поэтому числа для 128 немного отличаются от первого отчёта.",
              "- Контекст 128 переоценён из прежних весов, без нового обучения. Результаты и исходные веса первого опыта не перезаписаны. Его измерения скорости получены в предыдущем запуске на этом же компьютере.",
              "- Одинаковые hidden=128, layers=4, heads=4, local_window=8, AdamW lr=0.0006, weight_decay=0.01, clipping=1.0, FP32. Dropout и TF32 выключены, deterministic algorithms включены.",
              "- Внутри каждой длины число параметров трёх архитектур почти одинаково благодаря FFN 628/512/570. Между длинами число параметров растёт из-за таблицы позиционных embedding. Размеры FFN внутри опыта различаются, поэтому нельзя приписывать весь эффект именно 3D-координатам.",
              "- Для каждой длины и seed все архитектуры получают одинаковые обучающие окна. Порядок моделей меняется между seed. Между длинами окна и число независимых последовательностей в батче различаются, хотя число байт за шаг одинаково.",
              "- С ростом контекста оценочные блоки сбрасывают историю реже. Целевые байты одинаковы, но доступная история меняется — это часть проверки контекста.",
              "- Test не используется для остановки, подбора параметров или выбора бюджета. Бюджеты зафиксированы до основных запусков.", "",
              "## Ограничения", "",
              "Один английский роман, три seed и 45 секунд обучения на модель дают предварительное сравнение. Здесь не проверяется извлечение конкретного факта из далёкого прошлого и не измеряется качество диалога. Для такого вывода нужен отдельный тест на дальние зависимости. Сохранённые примеры свободной генерации следует рассматривать как проверку запуска; связность текста пока ограничена.", "",
              "Контрольный Transformer здесь небольшой и использует обучаемые абсолютные позиции. Конфигурации с RoPE, относительными смещениями или локальным attention не проверялись. Пространственные слои имеют заранее заданные локальные связи, что может помогать раннему обучению; чтобы установить роль именно геометрии, нужен контроль с тем же графом без 3D-координат.", "",
              "Увеличение контекста одновременно меняет длину окна, таблицу позиций и число независимых последовательностей в батче. Поэтому разница между 128 и 512 не изолирует единственный эффект добавления далёкой истории. Отдельный контроль без геометрии и другие корпуса по-прежнему нужны для вывода о пользе именно пространственных координат.", "",
              "## Файлы и воспроизведение", "",
              "Исходные измерения: `runs/context128_common/results.json`, `runs/context256_v1/results.json`, `runs/context512_v1/results.json`. Сводка со всеми исходными измерениями: `CONTEXT_RESULTS.json`. В каждом новом опыте сохранены веса двух состояний для каждой из девяти моделей, протокол, метрики и логи. Предыдущие результаты находятся в `runs/natural_v1`.", "",
              "В комплекте 21 тест: проверены причинность и обратное распространение при 512 байтах, общий оценочный префикс, сопоставимость сводки и прежние CPU/CUDA сценарии. Итоговый лог — `runs/context_checks/tests_final.txt`.", "",
              "```powershell", "Set-Location 'C:\\Users\\User\\Documents\\chatgpt\\testllm'",
              ".\\.venv\\Scripts\\python.exe -m spatiallm.context_eval --source runs/natural_v1 --out runs/context128_repeat --targets 70144 --device cuda",
              ".\\.venv\\Scripts\\python.exe -m spatiallm.experiment --seq-len 256 --batch-size 4 --eval-targets 70144 --steps 500 --seconds 45 --seeds 42 43 44 --out runs/context256_repeat --device cuda",
              ".\\.venv\\Scripts\\python.exe -m spatiallm.experiment --seq-len 512 --batch-size 2 --eval-targets 70144 --steps 500 --seconds 45 --seeds 42 43 44 --out runs/context512_repeat --device cuda", "```", "",
              "Запускайте команды последовательно и используйте новые каталоги. Для одинакового числа шагов эксперимент воспроизводим в том же окружении; за одинаковое время число шагов может немного измениться из-за нагрузки компьютера. Checkpoint-файлы опытов предназначены для генерации и оценки, не содержат optimizer/RNG для продолжения обучения.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default="CONTEXT_RESULTS.md")
    parser.add_argument("--json-out", default="CONTEXT_RESULTS.json")
    args = parser.parse_args()
    combined = compare(args.root)
    Path(args.out).write_text(report(combined), encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
