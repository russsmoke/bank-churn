import pandas as pd
import numpy as np
from faker import Faker

# Инициализация Faker для генерации реалистичных имен
fake = Faker()
Faker.seed(42)  # для воспроизводимости
np.random.seed(42)

# 1. Определяем количество синтетических записей
n_samples = 2000

# 2. Генерируем числовые признаки

# CreditScore (от 350 до 850)
credit_scores = np.random.randint(350, 851, n_samples)

# Age (от 18 до 80)
ages = np.random.randint(18, 81, n_samples)

# Tenure (от 0 до 10)
tenure = np.random.randint(0, 11, n_samples)

# Balance (две группы: часть клиентов имеют 0, часть - ненулевой баланс)
balance = np.zeros(n_samples)
non_zero_balance_indices = np.random.choice(n_samples, size=int(n_samples * 0.7), replace=False)
balance[non_zero_balance_indices] = np.random.uniform(10000, 250000, len(non_zero_balance_indices))
balance = np.round(balance, 2)

# NumOfProducts (1-4, с разной вероятностью)
num_of_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.5, 0.35, 0.1, 0.05])

# HasCrCard (70% имеют карту)
has_cr_card = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])

# IsActiveMember (50/50)
is_active_member = np.random.choice([0, 1], size=n_samples)

# EstimatedSalary (от 0 до 200,000)
estimated_salary = np.random.uniform(0, 200000, n_samples)
estimated_salary = np.round(estimated_salary, 2)

# 3. Генерируем категориальные и текстовые данные

# RowNumber (порядковый номер)
row_numbers = np.arange(1, n_samples + 1)

# CustomerId (8-значный ID)
customer_ids = np.random.randint(10000000, 99999999, n_samples)

# Surname (реалистичные фамилии)
surnames = [fake.last_name() for _ in range(n_samples)]

# Geography (распределение: France 50%, Germany 25%, Spain 25%)
geography = np.random.choice(['France', 'Germany', 'Spain'], size=n_samples, p=[0.5, 0.25, 0.25])

# Gender (50/50)
gender = np.random.choice(['Male', 'Female'], size=n_samples)

# 4. Собираем DataFrame (БЕЗ колонки 'Exited')
synthetic_df = pd.DataFrame({
    'RowNumber': row_numbers,
    'CustomerId': customer_ids,
    'Surname': surnames,
    'CreditScore': credit_scores,
    'Geography': geography,
    'Gender': gender,
    'Age': ages,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
})

# 5. Сохраняем в CSV
synthetic_df.to_csv(r'E:\bankChurnProject\bank-churn\main\data\synthetic_data.csv', index=False)

# Показываем информацию
print("Первые 10 строк синтетических данных (без целевой переменной):")
print(synthetic_df.head(10))
print("\n" + "="*80)
print("\nСтатистика по сгенерированным данным:")
print(synthetic_df.describe())
print("\n" + "="*80)
print(f"\nДатасет сохранен в файл: synthetic_data.csv")
print(f"Размер датасета: {synthetic_df.shape[0]} строк, {synthetic_df.shape[1]} колонок")
print("\nКатегориальные переменные:")
print(f"Geography: {synthetic_df['Geography'].value_counts().to_dict()}")
print(f"Gender: {synthetic_df['Gender'].value_counts().to_dict()}")
print(f"NumOfProducts: {synthetic_df['NumOfProducts'].value_counts().sort_index().to_dict()}")