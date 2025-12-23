import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import networkx as nx
import matplotlib.patches as patches

print("\n=== ЧАСТЬ 2: Расширенная модель с профилактикой и критическим состоянием ===")


class RiskAnalysis:
    def __init__(self):
        # Исправление: 7 состояний, чтобы соответствовать матрице 7x7
        self.states_extended = [
            'Нормальное (s0)',          # S0
            'Предаварийное (s1)',       # S1
            'Резервный режим (s2)',     # S2 
            'Восстановление (s3)',      # S3
            'Аварийное (s4)',           # S4
            'Профилактика (s5)',        # S5
            'Критическое (s6)'          # S6
        ]

        # Базовые интенсивности переходов (7x7)
        self.base_Q = np.array([
            [-0.22, 0.10, 0.00, 0.00, 0.00, 0.12, 0.00],  # s0 
            [0.06, -0.25, 0.10, 0.00, 0.02, 0.05, 0.02],  # s1
            [0.05, 0.04, -0.23, 0.07, 0.05, 0.00, 0.02],  # s2
            [0.30, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00],  # s3
            [0.00, 0.00, 0.30, 0.00, -0.35, 0.03, 0.02],  # s4
            [0.20, 0.05, 0.00, 0.00, 0.00, -0.25, 0.00],  # s5
            [0.00, 0.00, 0.00, 0.20, 0.05, 0.00, -0.25]   # s6
        ])

        # Ущерб для каждого состояния (в условных единицах) - 7 состояний
        self.damage = np.array([
            0,      # S0: Нормальное состояние - полная работоспособность (нет ущерба)
            5,      # S1: Предаварийное состояние - пониженная надежность (низкий ущерб)
            10,     # S2: Резервный режим - базовое ручное управление (умеренный ущерб)
            15,     # S3: Восстановление - ремонт, возврат в норму (затраты на ремонт)
            50,     # S4: Аварийное состояние (высокий ущерб)
            2,      # S5: Профилактика (незначительные затраты)
            100     # S6: Критическое состояние (максимальный ущерб)
        ])

    def runge_kutta_4(self, pi0, Q, t0, t_end, h):
        def f(pi):
            return np.dot(pi, Q)

        ts = np.arange(t0, t_end + h, h)
        pis = np.zeros((len(ts), len(pi0)))
        pis[0] = pi0

        for i in range(1, len(ts)):
            pi_current = pis[i - 1]
            k1 = f(pi_current)
            k2 = f(pi_current + h * k1 / 2)
            k3 = f(pi_current + h * k2 / 2)
            k4 = f(pi_current + h * k3)
            pi_next = pi_current + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            pi_next = np.maximum(pi_next, 0)
            pi_next /= pi_next.sum()
            pis[i] = pi_next

        return ts, pis

    def stationary_distribution(self, Q):
        n = Q.shape[0]
        A = np.vstack([Q.T[:-1], np.ones(n)])
        b = np.zeros(n)
        b[-1] = 1
        from numpy.linalg import lstsq
        pi_stat, _, _, _ = lstsq(A, b, rcond=None)
        return pi_stat, pi_stat.sum()

    def calculate_risk_levels(self, Q, risk_scenarios):
        """Расчет уровней риска для различных сценариев"""
        risk_results = {}

        for scenario_name, risk_factor in risk_scenarios.items():
            # Модифицируем матрицу переходов в соответствии с уровнем риска
            Q_modified = Q.copy()

            if risk_factor == 'high':
                # Высокий риск: увеличиваем переходы в критические состояния
                Q_modified[4, 6] *= 2  # Удваиваем переход из аварийного в критическое (s4→s6)
                Q_modified[1, 4] *= 1.5  # Увеличиваем переход из предаварийного в аварийное (s1→s4)
                Q_modified[0, 1] *= 1.2  # Увеличиваем переход из нормального в предаварийное (s0→s1)
            elif risk_factor == 'medium':
                # Средний риск: умеренные изменения
                Q_modified[4, 6] *= 1.5
                Q_modified[1, 4] *= 1.2
            else:  # no risk
                # Нет риска: уменьшаем опасные переходы
                Q_modified[4, 6] *= 0.5
                Q_modified[1, 4] *= 0.8
                Q_modified[0, 6] = 0.01  # Добавляем небольшой прямой риск из нормального в критическое

            # Корректируем диагональные элементы
            for i in range(len(Q_modified)):
                Q_modified[i, i] = -np.sum(Q_modified[i, :]) + Q_modified[i, i]

            # Вычисляем стационарное распределение
            pi_stat, _ = self.stationary_distribution(Q_modified)

            # Рассчитываем общий риск как сумму (вероятность * ущерб)
            total_risk = np.sum(pi_stat * self.damage)

            risk_results[scenario_name] = {
                'Q': Q_modified,
                'stationary': pi_stat,
                'total_risk': total_risk,
                'critical_prob': pi_stat[6],  # Вероятность критического состояния S6
                'failure_prob': pi_stat[4] + pi_stat[6]  # Вероятность аварийного + критического (S4 + S6)
            }

        return risk_results

    def create_risk_matrix(self, risk_results):
        """Создает карту риска в виде матрицы Вероятность × Ущерб"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Первая карта: общая визуализация рисков
        self._plot_risk_matrix(ax1, risk_results, "Общая карта рисков системы")

        # Вторая карта: детализированная по сценариям
        self._plot_detailed_risk_matrix(ax2, risk_results, "Детализированная карта рисков по сценариям")

        plt.tight_layout()
        plt.show()

    def _plot_risk_matrix(self, ax, risk_results, title):
        """Рисует основную карту рисков"""
        # Определяем зоны риска
        risk_zones = {
            'Высокий риск': {'prob_range': (0.3, 1.0), 'damage_range': (50, 100), 'color': 'red'},
            'Средний риск': {'prob_range': (0.1, 0.3), 'damage_range': (20, 50), 'color': 'orange'},
            'Низкий риск': {'prob_range': (0.05, 0.1), 'damage_range': (5, 20), 'color': 'yellow'},
            'Минимальный риск': {'prob_range': (0.0, 0.05), 'damage_range': (0, 5), 'color': 'green'}
        }

        # Рисуем зоны риска
        for zone_name, zone in risk_zones.items():
            prob_min, prob_max = zone['prob_range']
            damage_min, damage_max = zone['damage_range']
            rect = patches.Rectangle((damage_min, prob_min),
                                     damage_max - damage_min,
                                     prob_max - prob_min,
                                     linewidth=2, edgecolor='black',
                                     facecolor=zone['color'], alpha=0.3)
            ax.add_patch(rect)
            ax.text((damage_min + damage_max) / 2, (prob_min + prob_max) / 2,
                    zone_name, ha='center', va='center', fontweight='bold', fontsize=10)

        # Добавляем точки для каждого сценария
        colors = {'Высокий риск': 'darkred', 'Средний риск': 'darkorange', 'Риска нет': 'darkgreen'}
        markers = {'Высокий риск': 'X', 'Средний риск': 's', 'Риска нет': 'o'}

        for scenario, results in risk_results.items():
            prob = results['failure_prob']
            damage = results['total_risk']

            ax.scatter(damage, prob, s=200, c=colors.get(scenario, 'blue'),
                       marker=markers.get(scenario, 'o'), edgecolors='black', linewidth=2)

            # Подписи точек
            ax.annotate(scenario, (damage, prob),
                        xytext=(10, 10), textcoords='offset points',
                        fontweight='bold', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        ax.set_xlabel('УЩЕРБ (условные единицы)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ВЕРОЯТНОСТЬ', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.0)

    def _plot_detailed_risk_matrix(self, ax, risk_results, title):
        """Рисует детализированную карту рисков с разбивкой по состояниям"""
        # Создаем тепловую карту рисков по состояниям
        states_for_risk = ['Предаварийное', 'Аварийное', 'Критическое']
        scenarios = list(risk_results.keys())

        # Подготовка данных для тепловой карты
        risk_data = []
        for scenario in scenarios:
            results = risk_results[scenario]
            scenario_risks = [
                results['stationary'][1] * self.damage[1],  # S1: Предаварийное
                results['stationary'][4] * self.damage[4],  # S4: Аварийное
                results['stationary'][6] * self.damage[6]   # S6: Критическое
            ]
            risk_data.append(scenario_risks)

        risk_data = np.array(risk_data)

        # Создаем тепловую карту
        im = ax.imshow(risk_data, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')

        # Добавляем текст в ячейки
        for i in range(len(scenarios)):
            for j in range(len(states_for_risk)):
                text = ax.text(j, i, f'{risk_data[i, j]:.1f}',
                               ha="center", va="center", color="black", fontweight='bold')

        # Настройки осей
        ax.set_xticks(range(len(states_for_risk)))
        ax.set_yticks(range(len(scenarios)))
        ax.set_xticklabels(states_for_risk, rotation=45)
        ax.set_yticklabels(scenarios)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=ax, label='Уровень риска')

    def draw_comparison_graphs(self, Q_basic, Q_extended, states_basic, states_extended):
        """Рисует оба графа на одном дашборде"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Граф базовой модели
        G1 = nx.DiGraph()
        for i, s_from in enumerate(states_basic):
            G1.add_node(s_from)
            for j, s_to in enumerate(states_basic):
                if i != j and Q_basic[i, j] > 0:
                    G1.add_edge(s_from, s_to, weight=Q_basic[i, j])

        pos1 = nx.circular_layout(G1)
        nx.draw(G1, pos1, ax=ax1, with_labels=True, node_size=2000,
                node_color='lightblue', font_size=8, font_weight='bold')
        edge_labels1 = {(u, v): f"{d['weight']:.2f}" for u, v, d in G1.edges(data=True)}
        nx.draw_networkx_edge_labels(G1, pos1, ax=ax1, edge_labels=edge_labels1, font_color='red')
        ax1.set_title("Базовая модель (4 состояния)")

        # Граф расширенной модели
        G2 = nx.DiGraph()
        for i, s_from in enumerate(states_extended):
            G2.add_node(s_from)
            for j, s_to in enumerate(states_extended):
                if i != j and Q_extended[i, j] > 0:
                    G2.add_edge(s_from, s_to, weight=Q_extended[i, j])

        pos2 = nx.circular_layout(G2)
        nx.draw(G2, pos2, ax=ax2, with_labels=True, node_size=2000,
                node_color='lightgreen', font_size=8, font_weight='bold')
        edge_labels2 = {(u, v): f"{d['weight']:.2f}" for u, v, d in G2.edges(data=True)}
        nx.draw_networkx_edge_labels(G2, pos2, ax=ax2, edge_labels=edge_labels2, font_color='red')
        ax2.set_title("Расширенная модель (7 состояний)")

        plt.tight_layout()
        plt.show()

    def plot_risk_comparison(self, risk_results):
        """Визуализация сравнения рисков по уровням"""
        scenarios = list(risk_results.keys())
        total_risks = [risk_results[scen]['total_risk'] for scen in scenarios]
        critical_probs = [risk_results[scen]['critical_prob'] for scen in scenarios]
        failure_probs = [risk_results[scen]['failure_prob'] for scen in scenarios]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # График общего риска
        colors = ['red' if 'высокий' in scen.lower() else 'orange' if 'средний' in scen.lower() else 'green' for scen in
                  scenarios]
        bars1 = ax1.bar(scenarios, total_risks, color=colors, alpha=0.7)
        ax1.set_title('Общий риск по сценариям')
        ax1.set_ylabel('Величина риска')
        ax1.tick_params(axis='x', rotation=45)

        # Добавляем значения на столбцы
        for bar, value in zip(bars1, total_risks):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{value:.1f}', ha='center', va='bottom')

        # График вероятностей отказов
        x = np.arange(len(scenarios))
        width = 0.35
        bars2 = ax2.bar(x - width / 2, critical_probs, width, label='Критическое состояние', alpha=0.7)
        bars3 = ax2.bar(x + width / 2, failure_probs, width, label='Все отказы', alpha=0.7)
        ax2.set_title('Вероятности критических состояний и отказов')
        ax2.set_ylabel('Вероятность')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def analyze_performance_change(self, pi_basic, pi_extended):
        """Анализ изменения производительности системы"""
        # Производительность оцениваем как вероятность нахождения в работоспособных состояниях
        # Базовая: нормальное (0) + восстановление (3)
        performance_basic = pi_basic[0] + pi_basic[3]
        # Расширенная: нормальное (0) + восстановление (3) + профилактика (5)
        performance_extended = pi_extended[0] + pi_extended[3] + pi_extended[5]

        change = performance_extended - performance_basic
        change_percent = (change / performance_basic) * 100

        print(f"\nАнализ изменения производительности:")
        print(f"Производительность базовой модели: {performance_basic:.4f}")
        print(f"Производительность расширенной модели: {performance_extended:.4f}")
        print(f"Изменение производительности: {change:+.4f} ({change_percent:+.2f}%)")

        return change_percent


# Запуск анализа
risk_analyzer = RiskAnalysis()

# Определяем сценарии рисков
risk_scenarios = {
    'Высокий риск': 'high',
    'Средний риск': 'medium',
    'Риска нет': 'no'
}

# Анализ рисков для расширенной модели
risk_results = risk_analyzer.calculate_risk_levels(risk_analyzer.base_Q, risk_scenarios)

# Вывод результатов
print("\nРезультаты анализа рисков:")
print("Сценарий\t\tОбщий риск\tВер. крит. сост.\tВер. отказа")
for scenario, results in risk_results.items():
    print(
        f"{scenario}\t\t{results['total_risk']:.2f}\t\t{results['critical_prob']:.4f}\t\t{results['failure_prob']:.4f}")

# Визуализация карты рисков
risk_analyzer.create_risk_matrix(risk_results)

# Дополнительные графики сравнения
risk_analyzer.plot_risk_comparison(risk_results)

# Сравнение графов переходов
Q_basic_4x4 = np.array([
    [-0.20, 0.10, 0.05, 0.05],
    [0.15, -0.40, 0.15, 0.10],
    [0.05, 0.05, -0.25, 0.15],
    [0.10, 0.00, 0.05, -0.15]
])
states_basic = ['Нормальное (s1)', 'Предаварийное (s2)', 'Аварийное (s3)', 'Восстановление (s4)']

risk_analyzer.draw_comparison_graphs(Q_basic_4x4, risk_analyzer.base_Q,
                                     states_basic, risk_analyzer.states_extended)

# Анализ изменения производительности
pi_basic_stat, _ = risk_analyzer.stationary_distribution(Q_basic_4x4)
pi_extended_stat, _ = risk_analyzer.stationary_distribution(risk_analyzer.base_Q)

performance_change = risk_analyzer.analyze_performance_change(pi_basic_stat, pi_extended_stat)

# Дополнительная визуализация: динамика для среднего сценария риска
print("\nДинамика вероятностей для сценария 'Средний риск':")
Q_medium = risk_results['Средний риск']['Q']
# ИСПРАВЛЕНИЕ: начальный вектор для 7 состояний
pi0_extended = np.array([1, 0, 0, 0, 0, 0, 0])
times, probs_extended = risk_analyzer.runge_kutta_4(pi0_extended, Q_medium, 0, 20, 0.5)

plt.figure(figsize=(12, 8))
for i in range(len(risk_analyzer.states_extended)):
    plt.plot(times, probs_extended[:, i], marker='o', label=risk_analyzer.states_extended[i])
plt.xlabel('Время')
plt.ylabel('Вероятность')
plt.title('Динамика вероятностей состояний - Расширенная модель (Средний риск)')
plt.grid(True)
plt.legend()
plt.show()

print("\n=== Анализ завершен ===")
print("Карта рисков показывает распределение сценариев по матрице 'Вероятность × Ущерб'")
print("Красная зона: высокий риск - требует немедленного вмешательства")
print("Оранжевая зона: средний риск - требует планирования мер")
print("Зеленая зона: низкий риск - мониторинг")