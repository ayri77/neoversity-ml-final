# Прогнозування відтоку клієнтів і платформа ML-експериментів

[English version](README.md)

## Огляд проєкту

Проєкт розв'язує задачу прогнозування відтоку клієнтів телеком-компанії. Це
задача бінарної класифікації: для кожного клієнта потрібно визначити, чи належить
він до класу відтоку.

Конкурсні дані є анонімізованою високовимірною таблицею, що поєднує числові й
категоріальні ознаки та містить значну кількість пропущених значень. Навчальна
вибірка складається з **10 000 розмічених рядків**, тестова — з **2 500 рядків**,
а початковий простір містить **230 ознак**. Позитивний клас становить лише
**13,05%**, тому звичайна accuracy могла б приховувати низьку якість прогнозу для
меншості.

Основною метрикою конкурсу є **Balanced Accuracy** — середнє значення якості
розпізнавання обох класів. Вона дає змогу оцінювати модель з урахуванням
дисбалансу та не винагороджує стратегію, яка добре передбачає лише домінантний
негативний клас.

## Мета фінального проєкту

Робота має дві взаємопов'язані цілі:

1. отримати сильний і відтворюваний результат у задачі прогнозування відтоку;
2. побудувати повторно використовуваний workflow для систематичних
   ML-експериментів із табличними даними.

Друга ціль не була окремою формальною вимогою університетського завдання. Це
інженерне рішення, прийняте під час роботи, коли кількість версій даних, моделей,
параметрів, threshold і способів оцінювання стала занадто великою для надійного
керування лише через notebook.

## Етапи виконання

Робота розвивалася послідовно:

1. **Data audit.** Перевірено форму таблиць, типи ознак, цільову змінну,
   пропуски, константні колонки та узгодженість train/test.
2. **EDA.** Досліджено розподіли, cardinality категорій, дисбаланс класів,
   missingness і потенційні стратегії підготовки даних.
3. **Feature engineering.** Побудовано кілька версій підготовлених датасетів для
   перевірки гіпотез про пропуски, нульові значення та спільні патерни.
4. **Baseline models.** Окремо перевірено CatBoost, LightGBM і XGBoost.
5. **Порівняння та blending.** Проаналізовано OOF-прогнози моделей і
   зафіксовані лінійні суміші ймовірностей.
6. **Перехід до платформи.** Логіку датасетів, оцінювання, порівняння,
   threshold selection і артефактів перенесено у відтворювані модулі.
7. **Розширені експерименти.** Додано Research v2, Dataset Registry, Optuna,
   dataset comparison, deployment workflow та окремий AutoGluon screening.

Ця послідовність зберігає академічну історію дослідження, тоді як поточна
платформа забезпечує контрольоване продовження експериментів.

## Навігація по notebook

| Notebook | Призначення |
| --- | --- |
| [01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Первинна перевірка даних, схеми, цільової змінної та узгодженості train/test |
| [02_eda.ipynb](notebooks/02_eda.ipynb) | EDA числових і категоріальних ознак, missingness та cardinality |
| [03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb) | Побудова версій ознак і підготовлених Dataset Packages |
| [04_catboost_baseline.ipynb](notebooks/04_catboost_baseline.ipynb) | Baseline-експеримент із CatBoost |
| [05_lightgbm_baseline.ipynb](notebooks/05_lightgbm_baseline.ipynb) | Baseline-експеримент із LightGBM |
| [06_xgboost_baseline.ipynb](notebooks/06_xgboost_baseline.ipynb) | Baseline-експеримент із XGBoost |
| [07_model_comparison.ipynb](notebooks/07_model_comparison.ipynb) | Порівняння моделей та аналіз зафіксованого blend |
| [08_submission.ipynb](notebooks/08_submission.ipynb) | Формування й перевірка історичного submission на основі вибраних прогнозів |

Notebook 08 документує один із попередніх сценаріїв формування submission, але не
оголошується замороженим фінальним notebook для LMS.

<!-- FINAL NOTEBOOK TODO:
     Add the frozen final orchestration notebook after candidate selection.
-->

## Аналіз даних

Початковий аудит і EDA показали такі важливі властивості:

- train і test мають однаковий початковий набір із 230 вхідних ознак; цільова
  колонка присутня лише в навчальній вибірці;
- **209 із 230 ознак** мають хоча б один пропуск в обох вибірках;
- 154 ознаки мають щонайменше 75% пропущених значень, а **18 ознак повністю
  порожні**;
- у train виявлено **25 константних ознак**, включно з повністю порожніми;
  частина додаткових констант у test пояснюється дуже рідкісними значеннями;
- типи ознак неоднорідні: числові змінні поєднуються з 38 категоріальними
  полями;
- cardinality суттєво відрізняється: є бінарні, низькокардинальні та
  висококардинальні категорії з довгим хвостом;
- через частку позитивного класу 13,05% accuracy не є достатньою метрикою.

Пропуски розглядаються не лише як проблема якості даних, а й як потенційний
сигнал про стан клієнта або процес збору інформації. Саме ця гіпотеза стала
основою для кількох версій feature engineering.

## Підготовка ознак і версії датасетів

Кожна версія оформлена як Dataset Package з упорядкованими train/test ознаками,
цільовою змінною, manifest, hashes, row identity та lineage.

| Dataset ID | Ознак | Основна трансформація | Target dependency | Дослідницька мета |
| --- | ---: | --- | --- | --- |
| `v0_raw_minimal` | 205 | Видалення константних і повністю порожніх колонок | `none` | Мінімальний baseline на початкових корисних ознаках |
| `v1_missingness_summary` | 213 | Додавання сумарних counts/rates пропусків у рядку | `none` | Перевірка агрегованого missingness-сигналу |
| `v2_missingness_indicators` | 393 | Широкий набір індикаторів пропусків окремих ознак | `none` | Перевірка детальних патернів missingness |
| `v3_targeted_missingness` | 217 | Чотири вибрані індикатори пропусків | `exploratory` | Компактна перевірка target-informed індикаторів |
| `v4_zero_value_summary` | 209 | Counts/rates нульових значень у рядку | `none` | Перевірка агрегованого zero-value сигналу |
| `v5_joint_missingness_pattern` | 214 | Ознака спільного патерну пропусків | `none` | Опис комбінованого стану збору даних |
| `v6_compact_missingness_indicators` | 247 | Компактний набір структурно унікальних індикаторів | `none` | Зменшення надлишковості широкого набору |
| `v7_compact_zero_indicators` | 234 | Компактні індикатори нульових значень | `none` | Визначення ознак, де нульове значення несе сигнал |

`v3_targeted_missingness` має статус **exploratory**, оскільки його чотири
індикатори були вибрані з використанням target-informed evidence. Результати
цієї версії не змішуються з неупередженим рейтингом target-independent
датасетів.

## Чому була створена платформа

Після перших експериментів я дійшов висновку, що notebook добре підходять для
дослідження, але недостатні як єдине джерело стану проєкту. Кількість комбінацій
датасетів, моделей, seeds, threshold, preprocessing і гіперпараметрів швидко
зростала. Разом із цим зростали ризики порівнювати несумісні результати,
використати застарілий output або непомітно змінити протокол.

Попередній досвід хакатонів і ML-змагань також показав цінність workflow, який
можна перенести до наступної задачі. Тому ключовими вимогами стали:

- відтворюваність конфігурацій і результатів;
- контроль data leakage;
- чіткий provenance кожного запуску;
- незмінність історичних артефактів;
- об'єктивне порівняння лише сумісних експериментів;
- відокремлення швидкого screening від авторитетного оцінювання.

Повний технічний огляд архітектури наведено в [English README](README.md).

## Основні компоненти платформи

**Реалізовано:**

- Dataset Packages і Dataset Registry з manifest, schema/content hashes, lineage
  та row identity;
- Experiment Core / Research v2 для train-only оцінювання;
- model adapters для проєктних LightGBM, XGBoost і CatBoost;
- repeated і nested evaluation, OOF-прогнози та threshold calibration;
- same-dataset model comparison і identity-aware dataset comparison;
- leakage-aware оцінювання зафіксованого двокомпонентного blend;
- Optuna lifecycle для підтримуваних XGBoost/CatBoost кандидатів;
- підготовку deployment, автентифікацію конкурсних assets і локальну перевірку
  submission;
- Streamlit Control Panel як інтерфейс до дозволених CLI workflow;
- optional MLflow mirror для пошуку метаданих;
- crash-isolated standalone AutoGluon runner у відокремленому середовищі.

**Частково інтегровано:**

- відображення результатів великих campaign у Control Panel;
- перенесення вибраних AutoGluon кандидатів у first-class Research v2 adapters;
- узгоджений шлях від нових candidate predictions до фінального multi-model
  blending.

**Заплановано:**

- generalized cross-dataset і multi-model blending;
- розширене автоматичне формування звітів;
- узагальнення платформи для інших типів ML-задач.

Наявність реалізації runner або config не означає, що відповідний повний
експеримент уже виконано.

## Методологія оцінювання

Оцінювання ґрунтується на таких принципах:

- **stratified splits** зберігають частку класів;
- **repeated evaluation** показує стабільність результату між seeds;
- **nested evaluation** відокремлює threshold selection від зовнішнього
  validation fold;
- preprocessing, що навчається з використанням target, виконується лише всередині
  відповідного training partition;
- **OOF predictions** — прогнози для рядків, які не використовувалися для
  навчання відповідної моделі, — зберігаються з fold і row identity;
- threshold вибирається з training/OOF evidence після фіксації ймовірнісної
  моделі;
- deterministic seeds і hashes дають змогу відтворити assignments та перевірити
  сумісність запусків;
- competition test data не використовується для research evaluation.

Smoke, screening, exploratory, development, confirmation і final-deployment
evidence мають різні ролі. Метрики legacy cross-validation, AutoGluon validation,
Research v2, nested confirmation і Kaggle Public Score не можна трактувати як
однорідний leaderboard або безпосередньо порівнювати без урахування протоколу.

## Моделі

Проєктні шляхи моделювання охоплюють:

- **LightGBM** із fold-local OOF target encoding;
- **XGBoost** через numeric Research v2 adapter;
- **CatBoost** через numeric Research v2 adapter;
- зафіксовані лінійні blends сумісних OOF-ймовірностей;
- Optuna-supported пошук для XGBoost і CatBoost.

RealTabPFN, NeuralNetTorch та інші framework-managed кандидати розглядаються лише
в межах поточного AutoGluon screening. Вони не оголошуються фінальною моделлю.

## Роль AutoGluon

AutoGluon використовувався як допоміжний інструмент для масового скринінгу
сімейств моделей, способів підготовки ознак і фіксованих конфігурацій
гіперпараметрів.

AutoGluon операційно відокремлений від основного Research v2 workflow:

- використовує окреме середовище `.venv-autogluon`;
- запускається через standalone train-only runner;
- не підмінює Dataset Registry, Research v2, project-owned comparison або
  deployment logic;
- screening evidence не вважається автоматично авторитетним фінальним
  оцінюванням;
- перспективна конфігурація може бути заморожена й перевірена окремо;
- походження AutoGluon результатів указується явно.

Отже, проєкт розрізняє автоматичний portfolio screening, зафіксовану конфігурацію
кандидата та вручну реалізований project-owned pipeline. AutoGluon не подається
як уся сутність розв'язку.

## Поточні результати

Нижче наведено лише історичні результати з перевіреним локальним provenance.

| Тип evidence | Dataset | Кандидат | Threshold | Kaggle Public Score |
| --- | --- | --- | ---: | ---: |
| AutoGluon-screened frozen candidate | `v3_targeted_missingness` | `LightGBMPrep_r31` | 0.117 | **0.9112** |
| Project-owned manual model | `v3_targeted_missingness` | Manual LightGBM reproduction | 0.117 | 0.9023 |
| Project-owned manual blend | `v0_raw_minimal` | CatBoost 50% + XGBoost 50% | 0.130 | 0.8832 |

**0.9112** — це поточний найкращий перевірений історичний Kaggle Public Score,
а не оголошений заморожений фінальний submission. Він пов'язаний з
AutoGluon-screened кандидатом `LightGBMPrep_r31` на exploratory датасеті
`v3_targeted_missingness`.

Фінальний кандидат ще не зафіксований. Експерименти з кандидатами та підготовка
multi-model blending тривають, тому цей розділ буде повторно перевірено перед
фінальним поданням. Значення нещодавніх focused AutoGluon експериментів тут
навмисно не публікуються до спільного фінального оновлення.

<!-- FINAL RESULTS TODO:
     Reverify and update dataset, candidate, blend, threshold, score,
     submission identity, and date on 2026-08-02.
-->

## Відтворюваність

Відтворюваність забезпечується не лише seed, а повним набором ідентичностей:

- versioned YAML/JSON configs і resolved configs;
- Dataset Package manifests;
- schema, content, target і source hashes;
- train/test row identities та lineage;
- evaluation assignments і deterministic seeds;
- OOF artifacts, metrics і threshold summaries;
- унікальні no-overwrite run directories;
- terminal success/failure markers та artifact manifests.

Історичні filesystem artifacts є авторитетними й не перезаписуються за
замовчуванням. MLflow використовується лише як optional searchable mirror і не
перетворює локальний index на нове джерело істини.

Raw competition data, processed datasets, trained models, predictors, MLflow
state та submission CSV не додаються до Git за замовчуванням.

## Формування фінального прогнозу

Фінальний notebook має бути тонким orchestration entry point, а не копією всієї
платформи. Він повинен імпортувати протестовані модулі з `src/churn_ml` і:

1. документувати заморожену конфігурацію;
2. перевіряти вхідні файли, manifests, hashes і row identity;
3. навчати або виконувати вибраний зафіксований workflow;
4. виконувати full-data fit, якщо це визначено фінальним протоколом;
5. створювати рівно **2 500** прогнозів;
6. записувати колонки точно `index,y` у правильному порядку;
7. перевіряти binary labels і відсутність порушення row alignment;
8. повідомляти threshold, positive prediction count і SHA-256 submission.

Посилання на фінальний notebook буде додано лише після вибору кандидата,
clean-kernel Run All і перевірки тотожності з фактичним submission.

## Академічна доброчесність

У проєкті діють такі обмеження:

- labels конкурсного test не використовуються під час розроблення;
- відомі Orange labels не витягуються й не застосовуються до конкурсних test
  rows;
- конкурсні дані не розповсюджуються через репозиторій;
- `v3_targeted_missingness` явно позначено як exploratory;
- використання AutoGluon розкрито як auxiliary screening;
- існують project-owned реалізації моделей, evaluation, comparison, blending і
  deployment;
- Kaggle feedback не використовується як єдиний механізм вибору моделі, threshold
  або blend weights.

## Обмеження

Поточний стан має низку обмежень:

- розроблення ще триває, а фінальний кандидат не заморожений;
- у частині модулів залишаються припущення, специфічні для цієї конкуренції;
- legacy notebook-era evaluation і Research v2 поки співіснують;
- не всі AutoGluon кандидати інтегровані як first-class Research v2 adapters;
- повну заплановану dataset-by-model campaign не завершено;
- campaign-results UI та cross-dataset multi-blend залишаються неповними;
- окремі workloads потребують значного обсягу RAM і контрольованого запуску;
- screenshots, фінальний notebook і фінальна submission evidence ще очікують
  перевірки;
- актуальна позиція в leaderboard не заявляється.

## Подальший розвиток

Після завершення конкурсу можливі такі напрями:

- generic task і dataset schemas;
- адаптери для повторного використання в інших конкурсах;
- multiclass classification та regression;
- pluggable model workers і adapters;
- повні campaign orchestration та campaign reporting;
- generalized multi-model і cross-dataset blending;
- автоматичне формування відтворюваних звітів;
- CI-перевірки contracts, links і reproducibility;
- Docker packaging;
- remote immutable artifact storage.

Це майбутні напрями, а не твердження про вже завершену функціональність.

## Висновки

Дослідження показало, що пропуски й структура категоріальних ознак є важливою
частиною цієї задачі, а не лише технічною проблемою очищення даних. Дисбаланс
класів зробив Balanced Accuracy та контроль threshold центральними елементами
оцінювання.

Перехід від набору notebook до керованої платформи дав змогу фіксувати
Dataset Packages, evaluation protocols, OOF-прогнози, provenance і сумісність
порівнянь. Це зменшує ризик leakage та випадкового вибору результату з
несумісного протоколу.

Робота ще не завершена: фінальний кандидат, notebook і submission evidence мають
бути заморожені та повторно перевірені. Водночас створений workflow уже дає
основу для продовження досліджень після цієї конкуренції та адаптації до інших
табличних ML-задач.
