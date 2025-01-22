import matplotlib

matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt

# Константы
kB = 1.380658e-23  # Дж/К
hbar = 1.05457266e-34  # Дж·с
NA = 6.0221367e23  # моль^-1

# Параметры для Ниобия (Nb)
nb_params = {
    'kn': 8.0,
    'kp': 0.6802,
    'm': 92.9064 * 1.66055e-27,  # кг
    'ro': 2.8540e-10,  # м
    'D_kB': 21765.36,  # K
    'a': 2.53,
    'b': 9.34,
}


def calculate_Aw(kn, c, ro, a, b):
    K_R = hbar ** 2 / (kB * ro ** 2 * nb_params['m'])

    # Проверка на ноль
    if b == a:
        return np.nan  # или любое другое значение по умолчанию

    if c <= 0:
        return np.nan

    return K_R * (5 * kn * a * b * (b + 1)) / (144 * (b - a)) * (ro / c) ** (b + 2)


def calculate_theta(kn, V_ratio):
    c = ((6 * nb_params['kp'] * V_ratio) / (np.pi * NA)) ** (1 / 3)  # Расчет c на основе нормированного объема
    Aw = calculate_Aw(kn, c, nb_params['ro'], nb_params['a'], nb_params['b'])

    xi = 9 / kn
    if Aw <= 0 or xi <= 0:
        return np.nan
    term1 = -1 + np.sqrt(1 + (8 * nb_params['D_kB']) / (kB * Aw * xi ** 2))
    if term1 < 0:
        return np.nan
    return Aw * xi * term1


def calculate_gruneisen_params(V_ratio):
    theta = calculate_theta(nb_params['kn'], V_ratio)

    print(f"V_ratio: {V_ratio:.2f}, theta: {theta:.4e}")  # Отладочные сообщения

    if np.isnan(theta):
        return np.nan, np.nan, np.nan, np.nan

    X_w = nb_params['D_kB'] / theta
    gamma = -(nb_params['b'] + 2) / (6 * (1 + X_w))

    q = gamma * (X_w * (1 + 2 * X_w)) / (1 + X_w)

    z = gamma * (1 + 4 * X_w) - 2 * q

    return theta, gamma, q, z


# Интервал нормированного объема
V_ratios = np.linspace(0.01, 2, num=100)  # Изменено на [0.01, 2], чтобы избежать деления на ноль

# Списки для хранения результатов
theta_values_nb = []
gamma_values_nb = []
q_values_nb = []
z_values_nb = []

for V_ratio in V_ratios:
    theta, gamma, q, z = calculate_gruneisen_params(V_ratio)

    theta_values_nb.append(theta)
    gamma_values_nb.append(gamma)
    q_values_nb.append(q)
    z_values_nb.append(z)

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.plot(V_ratios, theta_values_nb)
plt.title('Температура Дебая Θ(V/V₀)')
plt.xlabel('V/V₀')
plt.ylabel('Θ (K)')

plt.subplot(222)
plt.plot(V_ratios, gamma_values_nb)
plt.title('Первый параметр Грюнайзена γ(V/V₀)')
plt.xlabel('V/V₀')
plt.ylabel('γ')

plt.subplot(223)
plt.plot(V_ratios, q_values_nb)
plt.title('Второй параметр Грюнайзена q(V/V₀)')
plt.xlabel('V/V₀')
plt.ylabel('q')

plt.subplot(224)
plt.plot(V_ratios, z_values_nb)
plt.title('Третий параметр Грюнайзена z(V/V₀)')
plt.xlabel('V/V₀')
plt.ylabel('z')

plt.tight_layout()
plt.savefig('niobiy_prop.png')
plt.close()
