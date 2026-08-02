# Повідомлення ментору

Вітаю! Надсилаю фінальний проєкт з прогнозування відтоку клієнтів:
https://github.com/ayri77/neoversity-ml-final

Фінальний виконаний notebook:
https://github.com/ayri77/neoversity-ml-final/blob/master/notebooks/09_final_project_report.ipynb

Найкращий перевірений Kaggle Public Score — **0,9112**. У проєкті реалізовано не
лише моделі, а й контрольовану ML-платформу: immutable Dataset Registry, Research
v2 з OOF та repeated validation, локальний Control Panel, row-aligned comparisons,
leakage-safe blending і перевірку готовності submission за схемою, порядком рядків,
кількістю позитивних прогнозів та SHA-256.

Dataset Package `v3_targeted_missingness` і залежні від нього кандидати явно
позначені як exploratory, оскільки набір ознак був target-informed. Kaggle Public
Score використано лише як зовнішній benchmark, а не для налаштування threshold чи
blend weights.

Через розмір і політику даних raw datasets, predictors, OOF/test probabilities та
інші великі artifacts не зберігаються в Git. У репозиторії є компактний
верифікований evidence bundle, фінальні графіки, технічний аудит і три рівні
інструкцій з відтворюваності.
