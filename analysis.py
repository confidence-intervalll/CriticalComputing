import matplotlib.pyplot as plt 
from scipy.special import comb 
import math
import numpy as np 
from typing import List, Tuple

class DiversityAnalyzer:
    """Класс для анализа диверсификации и надежности систем"""
    
    def __init__(self, versions: List[str]):
        self.versions = versions
        self.n = len(versions)
        self.length = len(versions[0]) if versions else 0
        
    @staticmethod
    def hamming_distance(version1: str, version2: str) -> int: 
        """Рассчет дистанции Хэмминга между двумя бинарными строками"""
        assert len(version1) == len(version2), "Длины версий должны совпадать" 
        return sum(c1 != c2 for c1, c2 in zip(version1, version2))
    
    @staticmethod
    def euclidean_distance(version1: str, version2: str) -> float:
        """Рассчет Евклидова расстояния (L2) между двумя бинарными строками"""
        assert len(version1) == len(version2), "Длины версий должны совпадать" 
        return math.sqrt(sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(version1, version2)))
    
    @staticmethod
    def manhattan_distance(version1: str, version2: str) -> int:
        """Рассчет Манхэттенского расстояния (L1) между двумя бинарными строками"""
        assert len(version1) == len(version2), "Длины версий должны совпадать"
        return sum(abs(int(c1) - int(c2)) for c1, c2 in zip(version1, version2))
    
    def _average_distance(self, distance_func) -> float:
        """Общий метод для расчета среднего расстояния"""
        if self.n < 2:
            return 0.0
            
        total_dist = 0
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                total_dist += distance_func(self.versions[i], self.versions[j])
                count += 1
        return total_dist / count
    
    def calculate_all_distances(self) -> dict:
        """Расчет всех метрик расстояния"""
        return {
            'hamming': self._average_distance(self.hamming_distance),
            'euclidean': self._average_distance(self.euclidean_distance),
            'manhattan': self._average_distance(self.manhattan_distance)
        }
    
    def adjusted_reliability(self, p: float, m: int, avg_dist: float, 
                           max_dist: float = None) -> Tuple[float, float]:
        """Расчет надежности с учётом средней диверсности версий"""
        if max_dist is None:
            max_dist = self.length
            
        # Коэффициент коррекции надежности от диверсности
        div_factor = 0.5 + 0.5 * (avg_dist / max_dist) 
        
        # Исправленная вероятность безотказной работы одной версии
        p_adj = min(p * div_factor, 1.0) 
        
        # Расчет вероятности безотказной работы системы (n, m)
        prob = 0 
        for k in range(m, self.n + 1): 
            prob += comb(self.n, k) * (p_adj ** k) * ((1 - p_adj) ** (self.n - k)) 
        return prob, p_adj
    
    def visualize_versions(self): 
        if not self.versions:
            return
            
        fig, ax = plt.subplots(self.n, 1, figsize=(self.length / 1.5, 1.5 * self.n)) 
        if self.n == 1: 
            ax = [ax] 
            
        for idx, version in enumerate(self.versions): 
            colors = ['green' if c == '1' else 'red' for c in version] 
            for i in range(self.length): 
                ax[idx].text(i, 0.5, version[i], fontsize=14, ha='center', 
                           va='center', color=colors[i]) 
            ax[idx].set_xlim(-1, self.length) 
            ax[idx].set_ylim(0, 1) 
            ax[idx].axis('off') 
            ax[idx].set_title(f"Версия {idx + 1}") 
        plt.tight_layout() 
        plt.show()
    
    def visualize_reliability_comparison(self, p: float, m: int):
        """Сравнительная визуализация надежности для всех метрик"""
        distances = self.calculate_all_distances()
        metrics = {
            'Хэмминг': distances['hamming'],
            'Евклид': distances['euclidean'],
            'Манхэттен': distances['manhattan']
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Сравнение расстояний
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        colors = ['blue', 'green', 'red']
        
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax1.set_ylabel('Среднее расстояние')
        ax1.set_title('Сравнение метрик расстояния')
        ax1.grid(True, alpha=0.3)
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # График 2: Сравнение надежности
        reliability_data = []
        for name, dist in metrics.items():
            sys_rel, adj_p = self.adjusted_reliability(p, m, dist, self.length)
            reliability_data.append((sys_rel, adj_p, name))
        
        sys_reliabilities = [x[0] for x in reliability_data]
        adj_ps = [x[1] for x in reliability_data]
        
        x_pos = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, sys_reliabilities, width, label='Надежность системы', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, adj_ps, width, label='Скорректированное p', alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names)
        ax2.set_ylabel('Вероятность')
        ax2.set_title('Сравнение надежности по разным метрикам')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Добавление подписей значений на столбцы надежности
        for bar, value in zip(bars1, sys_reliabilities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars2, adj_ps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return reliability_data

    def visualize_reliability(self, p: float, m: int, avg_dist: float, 
                            metric_name: str = "Метрика", max_dist: float = None):
        if max_dist is None:
            max_dist = self.length

        divs = np.linspace(0, max_dist, 300)
        reliabilities = []
        adjusted_ps = []
        for d in divs:
            rel, p_adj = self.adjusted_reliability(p, m, d, max_dist)
            reliabilities.append(rel)
            adjusted_ps.append(p_adj)

        fig, ax = plt.subplots(figsize=(10, 7))

        # График надежности системы
        ax.plot(divs, reliabilities, color='blue', label='Надежность системы', linewidth=2)
        ax.plot(divs, adjusted_ps, color='green', linestyle='dashed', 
                label='Корректированное p', linewidth=2)
        ax.set_xlabel(f'Средняя дистанция ({metric_name})', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Точки для стрелочек
        points = [0, avg_dist, max_dist]
        point_labels = ['Минимум', 'Текущее', 'Максимум']

        for i, x in enumerate(points):
            system_reliability, adjusted_p = self.adjusted_reliability(p, m, x, max_dist)
            
            # Стрелка для надежности (синяя сверху)
            ax.annotate(f"{system_reliability:.3f}", xy=(x, system_reliability), 
                        xytext=(x, system_reliability + 0.07),
                        textcoords='data',
                        ha='center', color='blue', fontweight='bold',
                        arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.7))
            
            # Стрелка для скорректированного p (зелёная снизу)
            ax.annotate(f"{adjusted_p:.3f}", xy=(x, adjusted_p), 
                        xytext=(x, adjusted_p - 0.12),
                        textcoords='data',
                        ha='center', color='green', fontweight='bold',
                        arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7))

            # Отметки точек
            ax.plot(x, system_reliability, 'bo', markersize=8)
            ax.plot(x, adjusted_p, 'go', markersize=8)
            

        ax.set_title(f'Зависимость надежности от диверсности ({metric_name})\n'
                    f'n={self.n}, m={m}, p={p}', fontsize=14, pad=20)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.2)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__": 
    versions = [ 
        "1101100001", 
        "1011110000", 
        "1011101111", 
    ] 
        
    analyzer = DiversityAnalyzer(versions)      
    analyzer.visualize_versions() 
        
    # Параметры системы
    base_p = 0.95 
    m_required = 2 
        
    # Расчет всех метрик
    distances = analyzer.calculate_all_distances()
        
    print("=== АНАЛИЗ ДИВЕРСНОСТИ СИСТЕМЫ ===")
    print(f"Количество версий: {analyzer.n}")
    print(f"Длина версий: {analyzer.length} бит")
    print(f"\nСРЕДНИЕ РАССТОЯНИЯ:")
    print(f"Хэмминг: {distances['hamming']:.4f} из {analyzer.length}")
    print(f"Евклид: {distances['euclidean']:.4f}")
    print(f"Манхэттен: {distances['manhattan']:.4f}")
        
    print(f"\n=== РАСЧЕТ НАДЕЖНОСТИ (p={base_p}, m={m_required}) ===")

    reliability_results = analyzer.visualize_reliability_comparison(base_p, m_required)
        
    print("\nРЕЗУЛЬТАТЫ НАДЕЖНОСТИ:")
    for sys_rel, adj_p, metric in reliability_results:
        print(f"{metric}:")
        print(f"  Скорректированное p: {adj_p:.4f}")
        print(f"  Надежность системы: {sys_rel:.4f}")
        

    analyzer.visualize_reliability(base_p, m_required, distances['hamming'], 
                                    "Хэмминг", analyzer.length)       
    analyzer.visualize_reliability(base_p, m_required, distances['euclidean'], 
                                    "Евклид", analyzer.length)
    analyzer.visualize_reliability(base_p, m_required, distances['manhattan'], 
                                    "Манхэттен", analyzer.length)