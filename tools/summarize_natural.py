"""Create the concise report from completed measurements, without extra packages."""
import argparse
import json
import math
from pathlib import Path


NAMES = {"spatial": "Пространственная", "transformer": "Transformer", "hybrid": "Гибрид"}


def build(results_path, manifest_path):
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = data["summary"]
    if len(data["records"]) != 18 or len(rows) != 6 or any(len(r["seeds"]) != 3 for r in rows):
        raise ValueError("report requires the completed three-model, three-seed experiment")
    protocol = data["protocol"]
    args = protocol["arguments"]
    lines = ["# Пространственная модель и гибрид: проверка на естественном тексте", "",
             "Локальный эксперимент завершён 9 ноября 2026 года на NVIDIA GeForce RTX 3050, 8 ГБ VRAM. Все модели обучены с нуля. Проверены три архитектуры и три seed (42, 43, 44), всего девять запусков и 18 сохранённых состояний.", "",
             "Источник: Jane Austen, *Pride and Prejudice*, [Project Gutenberg](https://www.gutenberg.org/ebooks/1342). Исходный текст с лицензией сохранён в проекте. Это один английский роман, а не универсальный языковой бенчмарк.", "",
             "## Основные результаты", "",
             "Ошибка — средняя negative log-likelihood на последних 10% романа, которые не используются для обучения. **Меньше — лучше.** Указаны среднее и стандартное отклонение по трём seed; это не доверительный интервал.", ""]
    for comparison, title in (("equal_steps", "Одинаковое число шагов — 500"),
                              ("equal_time", "Одинаковое время обучения — 45 секунд")):
        lines += [f"### {title}", "", "| Модель | Ошибка test, среднее ± SD | Ошибка validation | Шагов, среднее | Время обучения, с | Пик памяти, МиБ |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["comparison"] != comparison:
                continue
            lines.append(f"| {NAMES[row['architecture']]} | {row['test_loss_mean']:.4f} ± {row['test_loss_sd']:.4f} | {row['validation_loss_mean']:.4f} | {row['steps_mean']:.0f} | {row['training_seconds_mean']:.2f} | {row['peak_allocated_mib_mean']:.1f} |")
        lines.append("")
    time_rows = {r["architecture"]: r for r in rows if r["comparison"] == "equal_time"}
    baseline = time_rows["transformer"]
    lines += ["## Как читать результат", ""]
    for name in ("spatial", "hybrid"):
        candidate = time_rows[name]
        delta = candidate["test_loss_mean"] - baseline["test_loss_mean"]
        relative_perplexity = math.expm1(delta) * 100
        ratio = baseline["tokens_per_second_mean"] / candidate["tokens_per_second_mean"]
        paired = next(r for r in data["paired_deltas"] if r["architecture"] == name and r["comparison"] == "equal_time")
        wins = sum(d < 0 for d in paired["test_loss_delta_vs_transformer"])
        direction = "выше (хуже)" if delta > 0 else "ниже (лучше)"
        lines.append(f"- **{NAMES[name]}:** при одинаковом времени средняя test loss на {abs(delta):.4f} {direction}, чем у Transformer. Разница byte perplexity: {relative_perplexity:+.2f}%. Модель лучше Transformer в {wins} из 3 парных запусков. Transformer обрабатывает обучающие токены примерно в {ratio:.2f} раза быстрее.")
    best = min(time_rows.values(), key=lambda r: r["test_loss_mean"])
    lines += ["", f"Минимальная средняя ошибка в фиксированном 45-секундном бюджете — у модели **{NAMES[best['architecture']]}**. Это описательный результат этого опыта, а не доказательство общего превосходства архитектуры.", "",
              "Результат после одинакового количества шагов показывает, насколько эффективно модель использует одинаковое число обучающих примеров. Результат после одинакового времени учитывает, что более быстрая модель успевает обработать больше данных. Для практического вопроса об ускорении второй показатель важнее.", "",
              "## Полный протокол", "",
              "- Архитектуры: четыре пространственных слоя; четыре Transformer-слоя; гибрид spatial → Transformer → spatial → Transformer.",
              f"- Общие размеры: hidden={args['dim']}, layers={args['layers']}, heads={args['heads']}, context={args['seq_len']} UTF-8 байт, batch={args['batch_size']}.",
              "- Число параметров выровнено шириной FFN: SpatialLM 842368 (FFN 628), Transformer 842624 (FFN 512), гибрид 842496 (FFN 570). Расхождение менее 0.04%. Изменение FFN — дополнительное различие вариантов: по этому тесту нельзя изолировать пользу именно 3D-координат.",
              "- AdamW: learning rate 0.0006, weight decay 0.01, clipping 1.0. FP32, dropout 0, deterministic algorithms включены, TF32 выключен.",
              "- Для каждого seed обучающие окна одинаковы и подаются в одинаковом порядке. Порядок архитектур меняется между seed. Состояние для сравнения шагов снимается после 500 обновлений, то есть 512000 предсказанных байт на модель.",
              "- В 45 секунд входит сумма времени шагов: выбор батчей, передача на GPU, forward/backward, optimizer и синхронизация. Прогрев, валидация, запись файлов и генерация исключены. Порог может быть превышен на время одного шага. Это не полное время работы программы.",
              "- Деление текста — последовательные 80/10/10% по символам, без пересечения окон между частями. Размеры и SHA-256 ниже.",
              "- Оценка охватывает все полные непересекающиеся 128-байтовые блоки validation/test. Неполные хвосты (100 и 19 возможных целевых байт) отброшены одинаково для всех моделей. Test не используется для выбора настроек или остановки. Настройки и бюджеты после пилота не менялись.",
              "- Память — максимальная выделенная PyTorch память, не вся память драйвера/рабочего стола. Повторения по времени могут получить другое число шагов из-за текущей нагрузки компьютера.", "",
              "| Часть | Размер, байт | SHA-256 |", "|---|---:|---|"]
    for part, item in manifest["splits"].items():
        lines.append(f"| {part} | {item['bytes']} | `{item['sha256']}` |")
    lines += ["", "## Проверки и воспроизведение", "",
              "Все 16 тестов пройдены: прежние CPU/CUDA проверки плюс причинность и градиенты гибрида, почти равное число параметров, корректное взвешивание последнего batch при оценке и совместимость конфигурации. Продолжение из checkpoint без новых полей дополнительно проверяется в CPU-тесте resume. Итоговый лог находится в `runs/natural_v1/tests_final.txt`.", "",
              "```powershell", "Set-Location 'C:\\Users\\User\\Documents\\chatgpt\\testllm'",
              ".\\.venv\\Scripts\\python.exe -m spatiallm.experiment --device cuda --steps 500 --seconds 45 --seeds 42 43 44 --out runs/natural_repeat", "```", "",
              "Нужен новый каталог результатов. Подготовленные данные уже находятся в `data/natural`. Процедура, дополнительные параметры и команды генерации описаны в `EXPERIMENT_GUIDE.md`.", "",
              "Полные измерения: `runs/natural_v1/results.json`; протокол и окружение: `runs/natural_v1/protocol.json`. В каждом подкаталоге модели сохранены веса `equal_steps.pt`, `equal_time.pt`, метрики и пример генерации. Эти веса предназначены для оценки и генерации; состояние оптимизатора для продолжения в них не сохраняется. Исходные файлы предыдущей версии перед обновлением скопированы в `backups/natural_v1_20261109`.", "",
              "## Ограничения и следующий шаг", "",
              "Один роман, контекст 128 байт и три seed позволяют проверить работоспособность гипотезы, но не вынести вердикт обо всех пространственных языковых моделях. Бюджет короткий; соотношение результатов может измениться при долгом обучении. Проверенный пример генерации гибрида пока содержит несвязные фразы и искажённые слова: более низкая ошибка среди этих вариантов ещё не означает хорошего качества свободной генерации. Специальное GPU-ядро геометрии не реализовано. Отдельные гиперпараметры для каждой архитектуры не подбирались.", "",
              "Чтобы установить пользу именно геометрии, нужен следующий контроль: тот же разреженный граф и та же FFN, но веса связей без обучаемых 3D-координат. Для вывода об общей практической пользе также нужны более длинный контекст, другие книги/жанры и больший бюджет обучения. Текущий результат относится только к реализованным вариантам и измеренному режиму.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="runs/natural_v1/results.json")
    parser.add_argument("--manifest", default="data/natural/manifest.json")
    parser.add_argument("--out", default="NATURAL_RESULTS.md")
    args = parser.parse_args()
    Path(args.out).write_text(build(args.results, args.manifest), encoding="utf-8")
