# Code RAG Engine

Индексация кодовой базы через «технические паспорта» файлов (Gemini) и семантический поиск в ChromaDB. Точка входа — `main.py`.

## Установка

```bash
cd /path/to/code-rag-engine
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

В корне проекта создайте файл `.env`:

```env
GEMINI_API_KEY=ваш_ключ
# Список путей к проектам, которые индексируются и по которым ищется код.
# Разделитель — запятая. Индексы 0, 1, 2 … используются в флаге --project-index.
PROJECT_ROOTS=/path/to/repo-a,/path/to/repo-b
```

Векторная БД по умолчанию: `data/chroma/`. Промежуточные паспорта: каталог `pipeline_state/<имя_проекта>/passports/`.

---

## Общая справка

```bash
python main.py --help
python main.py <команда> --help
```

---

## Команды и примеры

### `summary` — генерация паспортов (саммари) по файлам

Без записи на диск, только план батчей:

```bash
python main.py summary --dry-run
```

Полный прогон для первого проекта в `PROJECT_ROOTS` (индекс по умолчанию `0`):

```bash
python main.py summary
```

Ограничить число батчей и задать параллелизм:

```bash
python main.py summary --batch-limit 3 --workers 2
```

Другой проект и свой каталог состояния:

```bash
python main.py summary --project-index 1 --state-dir my_state
```

### `embed` — загрузка паспортов в ChromaDB

Сначала должен существовать каталог паспортов (после `summary` или `process`). Пример:

```bash
python main.py embed
python main.py embed --project-index 1 --chunk-size 120 --state-dir pipeline_state
```

### `process` — полный конвейер: `summary` + `embed`

```bash
python main.py process
python main.py process --batch-limit 5 --workers 3 --chunk-size 120 --project-index 0
```

### `search` — семантический поиск по уже проиндексированным коллекциям

По всем проектам из `PROJECT_ROOTS` (аргумент `--project-index` не указан):

```bash
python main.py search "где обрабатывается оплата"
```

Только один проект:

```bash
python main.py search "authentication middleware" --project-index 0
```

Больше кандидатов на первом шаге и без расширения по доменам:

```bash
python main.py search "night delivery" --top-k 10 --no-expand
```

### `ask` — поиск + сбор контекста + отчёт в файл (полный RAG)

Нужны сохранённые паспорта (`summary` или `process`). По умолчанию учитываются все проекты; для одного укажите индекс.

```bash
python main.py ask "Как устроена ночная доставка?"
python main.py ask "What happens on buy button click?" --project-index 0 --top-k 8
python main.py ask "описание API заказов" --no-expand --output-dir reports
```

Отчёт сохраняется в указанную папку (по умолчанию `output/`) в виде markdown-файла.

---

## Типичный порядок работы

1. Настроить `PROJECT_ROOTS` и `.env`.
2. Один раз прогнать индексацию: `python main.py process` (или по шагам `summary`, затем `embed`).
3. Искать: `python main.py search "..."`.
4. Глубокий ответ с файлом: `python main.py ask "..."`.

После изменения кода в целевом репозитории имеет смысл снова запустить `summary`/`process` для актуализации паспортов и эмбеддингов.
