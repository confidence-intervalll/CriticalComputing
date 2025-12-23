import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.table import Table
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

print("=== КОМПЛЕКСНЫЙ АНАЛИЗ РИСКОВ СИСТЕМЫ ===")


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

    def stationary_distribution(self, Q):
        n = Q.shape[0]
        A = np.vstack([Q.T[:-1], np.ones(n)])
        b = np.zeros(n)
        b[-1] = 1
        pi_stat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pi_stat, pi_stat.sum()


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


def optimize_prevention_strategy(Q_base, max_prevention_rate, damage):
    prevention_strategies = []

    strategies = [
        {'name': 'Ранняя профилактика', 's1_to_s5': 0.7, 's2_to_s5': 0.3},
        {'name': 'Приоритет предаварийного', 's1_to_s5': 0.3, 's2_to_s5': 0.7},
        {'name': 'Равномерное распределение', 's1_to_s5': 0.5, 's2_to_s5': 0.5},
    ]

    risk_analyzer = RiskAnalysis()

    for strategy in strategies:
        Q_modified = Q_base.copy()

        total_prevention = strategy['s1_to_s5'] + strategy['s2_to_s5']
        scale_factor = max_prevention_rate / total_prevention

        Q_modified[0, 4] = strategy['s1_to_s5'] * scale_factor
        Q_modified[1, 4] = strategy['s2_to_s5'] * scale_factor

        for i in range(len(Q_modified)):
            Q_modified[i, i] = -np.sum(Q_modified[i, :]) + Q_modified[i, i]

        pi_stat, _ = risk_analyzer.stationary_distribution(Q_modified)
        total_risk = np.sum(pi_stat * damage)

        prevention_strategies.append({
            'name': strategy['name'],
            'Q': Q_modified,
            'stationary': pi_stat,
            'total_risk': total_risk,
            'prevention_distribution': {
                's1_to_s5': Q_modified[0, 4],
                's2_to_s5': Q_modified[1, 4]
            },
            'critical_prob': pi_stat[5],
            'failure_prob': pi_stat[2] + pi_stat[5],
            'performance': pi_stat[0] + pi_stat[3] + pi_stat[4]
        })

    return prevention_strategies


def plot_prevention_strategies_comprehensive(strategies, max_prevention_rate):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    names = [s['name'] for s in strategies]
    risks = [s['total_risk'] for s in strategies]
    critical_probs = [s['critical_prob'] for s in strategies]
    failure_probs = [s['failure_prob'] for s in strategies]
    performances = [s['performance'] for s in strategies]

    # График 1: Интегральный риск
    colors = ['red', 'orange', 'green']
    bars1 = ax1.bar(names, risks, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Сравнение интегрального риска стратегий', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Интегральный риск')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    for bar, risk in zip(bars1, risks):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{risk:.1f}', ha='center', va='bottom', fontweight='bold')

    # График 2: Вероятности отказов
    x = np.arange(len(names))
    width = 0.35
    bars2 = ax2.bar(x - width / 2, critical_probs, width, label='Критическое состояние',
                    color='red', alpha=0.7)
    bars3 = ax2.bar(x + width / 2, failure_probs, width, label='Все отказы',
                    color='orange', alpha=0.7)
    ax2.set_title('Вероятности критических состояний и отказов', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Вероятность')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Производительность
    bars4 = ax3.bar(names, performances, color=['lightblue', 'lightgreen', 'lightyellow'],
                    alpha=0.7, edgecolor='black')
    ax3.set_title('Производительность системы', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Вероятность работоспособности')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    for bar, perf in zip(bars4, performances):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')

    # График 4: Распределение ресурсов
    s1_prevention = [s['prevention_distribution']['s1_to_s5'] for s in strategies]
    s2_prevention = [s['prevention_distribution']['s2_to_s5'] for s in strategies]

    bars5 = ax4.bar(x - width / 2, s1_prevention, width, label='s1→s5 (Нормальное→Профилактика)',
                    color='blue', alpha=0.7)
    bars6 = ax4.bar(x + width / 2, s2_prevention, width, label='s2→s5 (Предаварийное→Профилактика)',
                    color='purple', alpha=0.7)
    ax4.set_title(f'Распределение ресурсов профилактики (лимит: {max_prevention_rate})',
                  fontsize=14, fontweight='bold')
    ax4.set_ylabel('Интенсивность перехода')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_strategy_radar_chart(strategies):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    categories = ['Снижение риска', 'Профилактика s1', 'Профилактика s2',
                  'Производительность', 'Снижение отказов']

    risks = [s['total_risk'] for s in strategies]
    s1_prev = [s['prevention_distribution']['s1_to_s5'] for s in strategies]
    s2_prev = [s['prevention_distribution']['s2_to_s5'] for s in strategies]
    perfs = [s['performance'] for s in strategies]
    failure_reduction = [1 - s['failure_prob'] for s in strategies]

    inverted_risks = [1 - (r - min(risks)) / (max(risks) - min(risks)) if max(risks) != min(risks) else 1 for r in
                      risks]

    normalized_data = []
    for i, strategy in enumerate(strategies):
        data = [
            inverted_risks[i],
            s1_prev[i] / max(s1_prev) if max(s1_prev) > 0 else 0,
            s2_prev[i] / max(s2_prev) if max(s2_prev) > 0 else 0,
            perfs[i],
            failure_reduction[i]
        ]
        normalized_data.append(data)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['red', 'orange', 'green']
    for i, strategy in enumerate(strategies):
        values = normalized_data[i] + normalized_data[i][:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title('Сравнение стратегий по multiple критериям', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.show()


def plot_prevention_effectiveness_surface(risk_analyzer, damage):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    s1_rates = np.linspace(0.01, 0.14, 15)
    s2_rates = np.linspace(0.01, 0.14, 15)
    S1, S2 = np.meshgrid(s1_rates, s2_rates)

    risks = np.zeros_like(S1)

    for i in range(S1.shape[0]):
        for j in range(S1.shape[1]):
            if S1[i, j] + S2[i, j] <= 0.15:
                Q_temp = risk_analyzer.base_Q.copy()
                Q_temp[0, 4] = S1[i, j]
                Q_temp[1, 4] = S2[i, j]

                for k in range(len(Q_temp)):
                    Q_temp[k, k] = -np.sum(Q_temp[k, :]) + Q_temp[k, k]

                pi_stat, _ = risk_analyzer.stationary_distribution(Q_temp)
                risks[i, j] = np.sum(pi_stat * damage)
            else:
                risks[i, j] = np.nan

    surf = ax.plot_surface(S1, S2, risks, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('Профилактика s1→s5')
    ax.set_ylabel('Профилактика s2→s5')
    ax.set_zlabel('Интегральный риск')
    ax.set_title('Поверхность эффективности профилактики', fontsize=14, fontweight='bold')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Уровень риска')
    plt.tight_layout()
    plt.show()


def student_task_comprehensive():
    risk_analyzer = RiskAnalysis()
    max_rate = 0.15
    strategies = optimize_prevention_strategy(risk_analyzer.base_Q, max_rate, risk_analyzer.damage)

    print("=" * 80)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ СТРАТЕГИИ ПРОФИЛАКТИКИ")
    print("=" * 80)
    print("Стратегия\t\t\tОбщий риск\tКрит. вер.\tОтказы\t\tПроизводит.\ts1→s5\t\ts2→s5")
    for strategy in strategies:
        prev = strategy['prevention_distribution']
        print(f"{strategy['name']}\t\t{strategy['total_risk']:.1f}\t\t{strategy['critical_prob']:.3f}\t\t"
              f"{strategy['failure_prob']:.3f}\t\t{strategy['performance']:.3f}\t\t"
              f"{prev['s1_to_s5']:.3f}\t\t{prev['s2_to_s5']:.3f}")

    plot_prevention_strategies_comprehensive(strategies, max_rate)
    plot_strategy_radar_chart(strategies)
    plot_prevention_effectiveness_surface(risk_analyzer, risk_analyzer.damage)

    best_strategy = min(strategies, key=lambda x: x['total_risk'])
    print(f"\n🎯 ОПТИМАЛЬНАЯ СТРАТЕГИЯ: {best_strategy['name']}")
    print(f"   Интегральный риск: {best_strategy['total_risk']:.1f}")
    print(f"   Производительность: {best_strategy['performance']:.3f}")


def advanced_analysis_with_visualization():
    risk_analyzer = RiskAnalysis()
    restriction_levels = [0.05, 0.10, 0.15, 0.20, 0.25]

    results = []
    for level in restriction_levels:
        strategies = optimize_prevention_strategy(risk_analyzer.base_Q, level, risk_analyzer.damage)
        best_strategy = min(strategies, key=lambda x: x['total_risk'])
        results.append({
            'restriction_level': level,
            'best_strategy': best_strategy['name'],
            'min_risk': best_strategy['total_risk'],
            'performance': best_strategy['performance']
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    levels = [r['restriction_level'] for r in results]
    risks = [r['min_risk'] for r in results]
    performances = [r['performance'] for r in results]
    strategies = [r['best_strategy'] for r in results]

    ax1.plot(levels, risks, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Уровень ограничения ресурсов профилактики')
    ax1.set_ylabel('Минимальный достижимый риск')
    ax1.set_title('Зависимость риска от доступных ресурсов', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    for i, (level, risk, strategy) in enumerate(zip(levels, risks, strategies)):
        ax1.annotate(strategy, (level, risk), xytext=(5, 5), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    ax2.plot(levels, performances, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Уровень ограничения ресурсов профилактики')
    ax2.set_ylabel('Производительность системы')
    ax2.set_title('Зависимость производительности от ресурсов', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


# ЗАПУСК ПРОГРАММЫ
if __name__ == "__main__":
    print("Запуск комплексного исследования стратегий профилактики...")
    student_task_comprehensive()

    print("\n" + "=" * 80)
    print("ПРОДВИНУТЫЙ АНАЛИЗ: Зависимость от уровня ограничений")
    print("=" * 80)
    advanced_results = advanced_analysis_with_visualization()

    print("\nВЫВОДЫ:")
    print("1. Сравнение стратегий показывает эффективность различных подходов к распределению ресурсов")
    print("2. 3D визуализация демонстрирует поверхность отклика системы на профилактические мероприятия")
    print("3. Радарная диаграмма позволяет оценить стратегии по multiple критериям")
    print("4. Анализ зависимости от ограничений помогает понять компромиссы при планировании ресурсов")