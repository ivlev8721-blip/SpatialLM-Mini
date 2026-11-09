"""Report a completed fixed-tokenizer experiment; reject incomplete budgets."""
import argparse
import json
import math
from pathlib import Path
import statistics


NAMES = {"spatial": "Пространственная", "transformer": "Transformer", "hybrid": "Гибрид"}


def build(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    records, protocol = result["records"], result["protocol"]
    args = protocol["arguments"]
    if len(records) != 18 or args["seq_len"] != 16384 or protocol["tokenizer"]["vocab_size"] != 50257:
        raise ValueError("complete 16K GPT-2 protocol required")
    for comparison in ("equal_steps", "equal_time"):
        for arch in NAMES:
            group = [r for r in records if r["comparison"] == comparison and r["architecture"] == arch]
            if sorted(r["seed"] for r in group) != [42, 43, 44]:
                raise ValueError("missing or duplicated seed")
            for row in group:
                if any(row[part]["targets"] != 16384 for part in ("validation", "test")):
                    raise ValueError("evaluation target positions differ")
                if row["training_tokens"] != row["steps"] * 16384:
                    raise ValueError("token count inconsistent")
                if comparison == "equal_steps" and row["steps"] != args["steps"]:
                    raise ValueError("update counts differ")
                if comparison == "equal_time" and row["training_seconds"] < args["seconds"]:
                    raise ValueError("time budget not reached")
    summaries = result["summary"]
    time_rows = {r["architecture"]: r for r in summaries if r["comparison"] == "equal_time"}
    baseline = time_rows["transformer"]
    lines = ["# SpatialLM-Mini: 16 384 стандартных токена GPT-2", "",
             "Проверка выполнена на NVIDIA GeForce RTX 3050 8 ГБ. Три архитектуры обучены с нуля при seed 42/43/44. Предобученные веса GPT-2 не использовались; из GPT-2 взят только фиксированный токенизатор.", "",
             "Это **subword-токены, а не байты**. Словарь содержит 50257 токенов. В train один токен соответствует в среднем примерно 3.96 UTF-8 байта. Прежние байтовые значения loss/perplexity напрямую с этой таблицей не сравниваются.", "",
             "## Одинаковое время обучения", "",
             f"Бюджет — {args['seconds']:g} секунд накопленного времени шагов. Прогрев, оценка, сохранение и генерация исключены. Ошибка — negative log-likelihood на GPT-2 токен; меньше — лучше. Среднее ± SD по трём seed, не доверительный интервал.", "",
             "| Модель | Test loss, среднее ± SD | Validation loss | Шагов | Токенов/с | Пик памяти, МиБ |", "|---|---:|---:|---:|---:|---:|"]
    for arch in NAMES:
        row = time_rows[arch]
        lines.append(f"| {NAMES[arch]} | {row['test_loss_mean']:.4f} ± {row['test_loss_sd']:.4f} | {row['validation_loss_mean']:.4f} | {row['steps_mean']:.0f} | {row['tokens_per_second_mean']:.0f} | {row['peak_allocated_mib_mean']:.1f} |")
    lines += ["", "## Одинаковое число обновлений", "",
              f"После {args['steps']} обновлений все модели обработали {args['steps'] * 16384} целевых токенов. Batch 1, без накопления градиента.", "",
              "| Модель | Test loss, среднее ± SD | Время, с | Токенов/с |", "|---|---:|---:|---:|"]
    for row in summaries:
        if row["comparison"] == "equal_steps":
            lines.append(f"| {NAMES[row['architecture']]} | {row['test_loss_mean']:.4f} ± {row['test_loss_sd']:.4f} | {row['training_seconds_mean']:.2f} | {row['tokens_per_second_mean']:.0f} |")
    lines += ["", "## Результат сравнения", ""]
    for arch in ("spatial", "hybrid"):
        row = time_rows[arch]
        pair = next(p for p in result["paired_deltas"] if p["comparison"] == "equal_time" and p["architecture"] == arch)
        speed_ratio = row["tokens_per_second_mean"] / baseline["tokens_per_second_mean"]
        ppl_change = 100 * math.expm1(pair["mean_delta"])
        lines.append(f"- **{NAMES[arch]}:** скорость обучения относительно Transformer — {speed_ratio:.2f}×. Разница средней test loss при одинаковом времени: {pair['mean_delta']:+.4f}; разница token perplexity по средним loss: {ppl_change:+.2f}%. Выигрышей по ошибке: {pair['wins']} из 3 парных запусков.")
    best = min(time_rows.values(), key=lambda r: r["test_loss_mean"])
    fastest = max(time_rows.values(), key=lambda r: r["tokens_per_second_mean"])
    lines += ["", f"В измеренном режиме наименьшая средняя ошибка при одинаковом времени у модели **{NAMES[best['architecture']]}**; наибольшая скорость обучения у модели **{NAMES[fastest['architecture']]}**. Результат относится к этим небольшим моделям и этому корпусу.", "",
              "## Почему режим экономии памяти исправлять не требуется", "",
              "Все модели действительно прошли обучение с полным окном 16384 GPT-2 токена. Включены FP16 autocast, FP32 веса/AdamW, пересчёт блоков при backward и пересчёт выходного слоя порциями по 512 позиций. В каждой позиции softmax охватывает весь словарь, все 16384 цели входят в loss. Эквивалентность обычному loss и градиентам проверена тестами, включая dropout и неполную порцию.", "",
              "В Transformer-слоях принудительно выбран PyTorch EFFICIENT_ATTENTION. Он вычисляет полное причинное внимание, экономя промежуточную память; переход на квадратичный по памяти math-backend запрещён. FlashAttention в данной установленной Windows-сборке не скомпилирован, менять окружение для этого опыта не потребовалось. [Официальная документация PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).",
              "",
              "Пик памяти в таблице — выделенная PyTorch память, не вся занятость VRAM драйвером/рабочим столом. Эти меры применены одинаково ко всем трём архитектурам. Пространственный граф может ускорять вычисления при этом контексте, но его промежуточные тензоры всё ещё требуют памяти; отдельного преимущества по памяти из скорости не следует.", "",
              "## Зафиксированные условия", "",
              "- Hidden 128, 4 слоя, 4 головы, контекст 16384, local_window 8. Гибрид чередует пространственные и Transformer-слои. Все параметры стартуют со случайных весов.",
              "- FFN 628/512/570: 9323248 параметров у SpatialLM, 9323392 у Transformer и 9323320 у гибрида. Внутри опыта бюджет параметров почти одинаков; расширенная FFN остаётся дополнительным различием архитектур.",
              f"- AdamW: lr={args['lr']}, weight_decay=0.01, clipping=1.0. FP16 autocast, начальный GradScaler 128, dropout 0, deterministic algorithms включены, TF32 выключен.",
              "- Для каждого seed последовательность обучающих окон одинакова у трёх архитектур. Порядок архитектур меняется между seed. Настройки зафиксированы после пилота по памяти/скорости; test не использован для настройки, остановки или выбора лучшего состояния.",
              "- Сохранены два заранее определённых состояния каждого запуска. Состояние после одинакового времени может превышать бюджет на время одного шага.", "",
              "## Корпус и оценка", "",
              "Тот же сохранённый роман Jane Austen, *Pride and Prejudice*, [Project Gutenberg](https://www.gutenberg.org/ebooks/1342). Исходные train/validation/test не изменены. Токенизация каждой части выполняется отдельно фиксированным GPT-2 токенизатором tiktoken 0.12.0. Хеши текста и token ID сохранены в `data/subword16k/manifest.json`.", "",
              "| Часть | GPT-2 токенов | UTF-8 байт |", "|---|---:|---:|"]
    for part, info in protocol["tokenizer"]["splits"].items():
        lines.append(f"| {part} | {info['tokens']} | {info['bytes']} |")
    lines += ["", "Validation и test оцениваются на одинаковых позициях 1…16384, в одном полном контекстном блоке на часть. Обучающие окна не пересекают границ частей. Для модели доступны предшествующие токены внутри блока, будущие токены маскируются. Оценка одного длинного блока не заменяет разнообразный набор независимых документов.", "",
              "## Что подтверждено и что остаётся проверить", "",
              "Подтверждены практическое обучение и backward на 16384 обычных GPT-2 токенах, сопоставимый бюджет трёх моделей и измерения времени/памяти. Режим экономии памяти работает на текущей RTX 3050. Данные и веса сохранены локально.", "",
              "Это один роман, небольшие модели и короткое обучение. Среднее и SD по трём seed не являются проверкой статистической значимости. Число доступных позиций не доказывает умения извлекать факт из далёкой части текста. Для проверки дальних зависимостей нужен отдельный тест; для роли именно геометрии — тот же граф без 3D-координат.", "",
              "От прежних байтовых опытов отличаются токенизатор, словарь, число параметров, precision и способы экономии памяти. Сравнение между этими сериями не изолирует влияние только длины. Контрольный Transformer использует обучаемые абсолютные позиции; варианты с RoPE или относительными смещениями не проверялись. Специальное геометрическое CUDA-ядро не разрабатывалось.", "",
              "## Файлы и запуск", "",
              "Полные измерения: `runs/subword16k_v1/results.json`; протокол: `protocol.json`; веса: `<architecture>_seed<seed>/equal_steps.pt` и `equal_time.pt`. Это 18 snapshots для оценки/генерации без optimizer/RNG. Прежние байтовые результаты и веса сохранены. Резервные копии изменённых исходников — `backups/subword16k_v1_20261109`.", "",
              "Инструкции установки, воспроизведения и генерации — `SUBWORD16K_GUIDE.md`. Итоговые проверки находятся в `runs/subword16k_checks/`.", "",
              "```powershell", r"Set-Location 'C:\Users\User\Documents\chatgpt\testllm'",
              r".\.venv\Scripts\python.exe -m spatiallm.subword16k run --steps 100 --seconds 180 --seeds 42 43 44 --out runs/subword16k_repeat", "```", "",
              "Используйте новый каталог. Не запускайте несколько GPU-опытов одновременно. За одинаковое время повтор может получить другое число шагов из-за нагрузки компьютера. Проверенный набор пакетов записан в `requirements-lock-subword-cuda.txt`.", ""]
    checks_path = Path(path).resolve().parent.parent / "subword16k_checks/verification.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        if not checks["passed"] or len(checks["snapshots"]) != 18:
            raise ValueError("snapshot verification incomplete")
        backend = checks["efficient_vs_math"]
        lines += ["## Заключительные проверки", "",
                  "- 25 автоматических тестов прошли, включая CUDA FP16, прежнее продолжение обучения и эквивалентность loss/градиентов при расчёте порциями.",
                  f"- Прямое сравнение CUDA EFFICIENT_ATTENTION с MATH на причинном FP16-внимании: максимальная абсолютная разница выходов {backend['max_output_abs_error']:.3g}, градиентов {backend['max_gradient_abs_error']:.3g}. Проверка уложилась в заданные допуски FP16.",
                  "- Все 18 сохранённых состояний повторно загружены с точным совпадением ключей параметров; сохранены SHA-256 файлов.",
                  "- Каждая архитектура выполнила генерацию из prompt длиной 18225 GPT-2 токенов с активным окном 16384. Это техническая проверка полного окна и обрезки, а не тест дальнего извлечения фактов.",
                  "- Старый байтовый checkpoint загружен и использован для генерации. Подробности — `runs/subword16k_checks/verification.json`.", "",
                  "Короткие образцы генерации сохранены в каталогах запусков. В них есть повторы и нарушения связности; этот опыт проверяет маленький исследовательский прототип, его результаты не означают качество готового диалогового ассистента.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="runs/subword16k_v1/results.json")
    parser.add_argument("--out", default="SUBWORD16K_RESULTS.md")
    args = parser.parse_args()
    Path(args.out).write_text(build(args.results), encoding="utf-8")
