### Оптимизационный метод
- Реализация этого метода представлена в `optimization.py`. 
- Функция `optimize_find_path` ищет оптимальную траекторию пути 
для заданной сетки.

### Алгоритм A*
- Реализация этого метода представлена в `a_star.py`. 
- Функция `a_star_find_path` ищет оптимальную траекторию пути 
для заданной сетки. 
- `data_draw_paths` проецирует пути для различных размеров сеток на поверхность затрат на укладку 
дорожного полотна.

### Метод распространения волнового фронта
- Реализация этого метода представлена в `fmm.py`. 
- Функция `fmm_find_path` ищет оптимальную траекторию пути 
для заданной сетки. 
- `data_draw_paths` проецирует пути для различных размеров сеток на поверхность затрат на укладку 
дорожного полотна.

### Метод быстро растущих случайных деревьев
- Реализация этого метода представлена в `trrt.py`. 
- Функция `trrt_find_path` ищет оптимальную траекторию пути 
для заданной сетки. 
- `data_draw_paths` проецирует пути для различных размеров сеток на поверхность затрат на укладку 
дорожного полотна. 
- `show_tree` отрисовывает дерево, которое было получено в результате работы метода.

### Сравнение методов
- Сравнение представлено в файле `comparision.py`.