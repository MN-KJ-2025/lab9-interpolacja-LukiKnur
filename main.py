# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    
    if not (isinstance(n, (int, np.integer)) and n > 0):
        return None
    
    if n == 1:
        return np.array([1.0])

    nodes = np.ones(n)

    for k in range(len(nodes)):
        nodes[k] = np.cos(k * np.pi / (n - 1))

    return nodes


    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        return None

    if n == 1:
        return np.array([1.0])

    weights = np.ones(n)

    for k in range(n):
        weights[k] = (-1) ** k

    weights[0] *= 0.5
    weights[-1] *= 0.5

    return weights


def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not all(isinstance(arr, np.ndarray) for arr in (xi, yi, wi, x)):
        return None

    if xi.ndim != 1 or yi.ndim != 1 or wi.ndim != 1 or x.ndim != 1:
        return None

    if len(xi) == 0 or len(xi) != len(yi) or len(xi) != len(wi):
        return None


    result = np.zeros_like(x, dtype=float)

    for j in range(len(x)):
      
        diff = x[j] - xi
        zero_idx = np.where(np.isclose(diff, 0.0))[0]

        if zero_idx.size > 0:
            result[j] = yi[zero_idx[0]]
        else:
            numerator = np.sum(wi * yi / diff)
            denominator = np.sum(wi / diff)
            result[j] = numerator / denominator

    return result


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if isinstance(xr, (int, float, np.integer, np.floating)) and \
       isinstance(x,  (int, float, np.integer, np.floating)):
        return float(abs(xr - x))

 
    try:
        xr_arr = np.asarray(xr, dtype=float)
        x_arr = np.asarray(x, dtype=float)
    except Exception:
        return None

   
    if xr_arr.shape != x_arr.shape:
        return None

    if xr_arr.size == 0:
        return None

    
    return float(np.max(np.abs(xr_arr - x_arr)))
