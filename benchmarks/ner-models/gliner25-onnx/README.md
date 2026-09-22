# GLiNER2.5 ONNX: фиксированное сравнение с native PyTorch

Экспериментальная ветка `experiment/gliner25-onnx` от GLiNER `1fffea1`.
Проверяется [`DanKau/gliner2.5-multi-v1-onnx`](https://huggingface.co/DanKau/gliner2.5-multi-v1-onnx/tree/481ad683a5420349c20f7ccc992efd25f9f3b809)
revision `481ad683a5420349c20f7ccc992efd25f9f3b809`.
Production и существующие ZIP/контейнеры не переключаются на этот backend.

## Протокол

Те же 5095 текстов: organizer 446, HiveTrace PII Bench 1810,
Red Mad Robot 2839 с прежними двумя исключениями ошибок BIO-выравнивания.
Модель, данные, порядок, правила и оценщики фиксируются до inference.
Настройки из выбранного GLiNER: `described-names`, порог 0.8, FP32,
batch=1, 4 CPU threads, seed=20260922; TF32 отключён.
Успешный replay прежних baseline-агрегатов обязателен перед запуском.

Сравниваются полные корпуса и все исходные gold-типы. Маскирование считается
по буквенно-цифровым позициям; typed span/character дополнительно штрафуют
неверные классы и границы. Organizer содержит прежнюю предварительную AI-разметку,
а не официальный ground truth. Эти корпуса уже участвовали в разработке правил,
а схема GLiNER выбиралась на organizer; это не слепой тест всего решения.
ONNX-настройки по результатам качества не подбираются.

## Происхождение и адаптер

Экспорт указан как полученный из fastino revision `aaecfe45…`; наш reference —
`a221b77…`. Все пять native model/config/tokenizer-файлов этих revisions
эквивалентны по HF metadata. Отдельный [provenance.json](provenance.json)
содержит проверенные идентификаторы. ONNX tokenizer.json также совпадает.

`seif/gliner_onnx.py` заменяет encoder и boundary head на ONNX Runtime.
Остаются исходные GLiNER processor, schema descriptions, word/query pooling,
abstention, выбор непересекающихся spans и объединение длинных окон.
Это важно: упрощённая инструкция ONNX-карты не описывает все детали native
декодирования. Наше `flat` использует максимум суммы confidence интервалов,
а не жадный выбор одного максимального span.

Экспериментальный адаптер **по-прежнему требует Torch и native checkpoint**
для исходного runtime/препроцессинга. Нейросетевые encoder/boundary вычисления
идут через ORT; это не готовая поставка без PyTorch и не Java-реализация автора.
FP16, INT8, TensorRT и batching в этот фиксированный эксперимент не входят.

Зависимости: [`requirements-gliner-onnx.txt`](../../../deploy/ner/requirements-gliner-onnx.txt).
ONNX Runtime GPU 1.30.0, ONNX 1.23.0, GLiNER2 2.0.0, Torch 2.14.0+cu130,
Transformers 4.57.6, Python 3.13.7. GPU — RTX 4070 Ti SUPER.
Локальные файлы обязательны, автоматических загрузок в inference нет.

## Воспроизведение

Создать Python 3.13 среду и установить lock-файл выше. Скачать pinned snapshot
ONNX и прежний pinned `fastino/gliner2.5-multi-v1@a221b77a8baf4a613b8f8652661d41fa10a5641e`.
`PUBLIC_RUN_DIR` — frozen public run из [GLiNER эксперимента](../gliner25-multi-v1/README.md).
`ONNX_PATH` и `NATIVE_PATH` — локальные snapshot-каталоги, `RUN_DIR` — новый
каталог внутри игнорируемого `local-data/` этого checkout.

```bash
.venv/bin/python scripts/compare_onnx.py prepare \
  --run-dir "$RUN_DIR" --public-run-dir "$PUBLIC_RUN_DIR" \
  --model-path "$ONNX_PATH" --native-path "$NATIVE_PATH"
.venv/bin/python scripts/compare_onnx.py cache \
  --run-dir "$RUN_DIR" --public-run-dir "$PUBLIC_RUN_DIR" \
  --model-path "$ONNX_PATH" --native-path "$NATIVE_PATH" \
  --backend onnx --device cuda --scope all
.venv/bin/python scripts/compare_onnx.py evaluate \
  --run-dir "$RUN_DIR" --public-run-dir "$PUBLIC_RUN_DIR" \
  --model-path "$ONNX_PATH" --native-path "$NATIVE_PATH" \
  --backend onnx --device cuda --scope all --output "$RUN_DIR/comparison-gpu.json"
```

Для CPU используются `--device cpu --scope organizer`. Измерения разных
моделей/устройств выполняются последовательно, без конкурирующего inference.
Повторная запись выходных файлов запрещена. Исходные тексты и model weights
не попадают в новые Git-артефакты.
