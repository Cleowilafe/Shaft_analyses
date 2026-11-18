from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


# ==========================================================
# 1. PAINEL DE CONTROLE (EDITE APENAS AQUI)
# ==========================================================

# --- Chaves de Execução e Visualização ---
PLOT = True                     # Chave geral: True exibe gráficos, False desativa todos
PLOT_PLANOS = True              # Plota momentos fletores nos planos XY e XZ
PLOT_RESULTANTE_TORQUE = True   # Plota momento fletor resultante (M_R) e torque (T)
PLOT_DEFLEXAO = True            # Plota deflexão resultante e destaca os pontos de carga
CALCULAR_RAYLEIGH = True        # Calcula frequência natural de flexão (Rayleigh)
CALCULAR_TORCIONAL = True       # Calcula frequência natural torcional (pinhão-coroa)
IMPRIMIR_RESUMO = True          # Exibe reações, máximos, deflexões e frequências naturais
SALVAR_FIGURAS = False          # Salva os gráficos gerados em arquivos PNG
PASTA_SAIDA = "resultados"      # Pasta onde as figuras serão salvas (se SALVAR_FIGURAS=True)

# --- Propriedades do Material e Geometria [Pa, m] ---
E = 200e9                       # Módulo de elasticidade [Pa]
G = 79.3e9                      # Módulo de cisalhamento do aço [Pa] (Shigley)
D_MENOR = 0.02                  # Diâmetro menor [m]
D_MAIOR = 0.05                  # Diâmetro maior [m]
TRECHOS_L = (0.070, 0.030, 0.032, 0.028, 0.064)  # (L1, L2, L3, L4, L5) [m]

# --- Carregamentos [m, N, N·m] ---
NOMES_CARGAS = ["Pinhão", "Coroa"]          # Rótulos dos pontos de aplicação
POSICOES_CARGAS = [0.06406, 0.11148]        # [x_pinhao, x_coroa] [m]
FORCAS_Y = [2871.675, -3361.848]            # Cargas no plano XY [N]
FORCAS_Z = [-1578.707, 88.033]              # Cargas no plano XZ [N]
TORQUE_TRANSMISSAO = 238.91                 # Torque entre pinhão e coroa [N·m]

# --- Massas e Raios Primitivos (Flexão Rayleigh e Torção) [kg, m, m/s²] ---
MASSAS_CARGAS = [0.1, 0.60]      # [massa_pinhao, massa_coroa] [kg]
RAIOS_CARGAS = [0.040, 0.080]   # [r_pinhao, r_coroa] [m] -> AJUSTAR COM SEU PROJETO
GRAVIDADE = 9.81                # Aceleração da gravidade [m/s²]

# --- Parâmetros Numéricos ---
N_GAUSS = 8                     # Pontos de quadratura de Gauss-Legendre por trecho
N_PONTOS = 5001                 # Resolução da malha para gráficos e máximos


# ==========================================================
# 2. ESTRUTURAS DE CONFIGURAÇÃO
# ==========================================================

@dataclass
class PainelControle:
    """Agrupa as opções de execução, saída no terminal e plotagem."""
    plot: bool = PLOT
    plot_planos: bool = PLOT_PLANOS
    plot_resultante_torque: bool = PLOT_RESULTANTE_TORQUE
    plot_deflexao: bool = PLOT_DEFLEXAO
    calcular_rayleigh: bool = CALCULAR_RAYLEIGH
    calcular_torcional: bool = CALCULAR_TORCIONAL
    imprimir_resumo: bool = IMPRIMIR_RESUMO
    salvar_figuras: bool = SALVAR_FIGURAS
    pasta_saida: str = PASTA_SAIDA


@dataclass
class ConfiguracaoEixo:
    """Armazena os parâmetros físicos e numéricos do eixo."""
    E: float = E
    G: float = G
    d_menor: float = D_MENOR
    d_maior: float = D_MAIOR
    trechos_L: tuple[float, float, float, float, float] = TRECHOS_L
    nomes_cargas: list[str] = field(default_factory=lambda: list(NOMES_CARGAS))
    posicoes: np.ndarray = field(default_factory=lambda: np.array(POSICOES_CARGAS, dtype=float))
    Fy: np.ndarray = field(default_factory=lambda: np.array(FORCAS_Y, dtype=float))
    Fz: np.ndarray = field(default_factory=lambda: np.array(FORCAS_Z, dtype=float))
    T_transmissao: float = TORQUE_TRANSMISSAO
    massas: np.ndarray = field(default_factory=lambda: np.array(MASSAS_CARGAS, dtype=float))
    raios: np.ndarray = field(default_factory=lambda: np.array(RAIOS_CARGAS, dtype=float))
    g: float = GRAVIDADE
    n_gauss: int = N_GAUSS
    n_pontos: int = N_PONTOS

    def __post_init__(self) -> None:
        self.L1, self.L2, self.L3, self.L4, self.L5 = self.trechos_L
        self.x1 = self.L1
        self.x2 = self.x1 + self.L2
        self.x3 = self.x2 + self.L3
        self.x4 = self.x3 + self.L4
        self.x5 = self.x4 + self.L5
        self.L = self.x5

        if len(self.nomes_cargas) != len(self.posicoes):
            self.nomes_cargas = [f"Carga {i + 1}" for i in range(len(self.posicoes))]

        marcos = [0.0, self.x1, self.x2, self.x3, self.x4, self.x5]
        self.pontos_descontinuidade = sorted(set(marcos + list(self.posicoes)))


# ==========================================================
# 3. GEOMETRIA DA SEÇÃO TRANSVERSAL
# ==========================================================

def calcular_diametro(x: float, cfg: ConfiguracaoEixo) -> float:
    """Retorna o diâmetro do eixo [m] na posição axial x [m]."""
    if x <= cfg.x1:
        return cfg.d_menor
    elif x <= cfg.x2:
        return cfg.d_menor + ((cfg.d_maior - cfg.d_menor) * (x - cfg.x1) / cfg.L2)
    elif x <= cfg.x3:
        return cfg.d_maior
    elif x <= cfg.x4:
        return cfg.d_maior - ((cfg.d_maior - cfg.d_menor) * (x - cfg.x3) / cfg.L4)
    return cfg.d_menor


def calcular_inercia(x: float, cfg: ConfiguracaoEixo) -> float:
    """Retorna o momento de inércia de área [m^4] na posição axial x [m]."""
    d = calcular_diametro(x, cfg)
    return np.pi * d**4 / 64.0


def calcular_momento_polar(x: float, cfg: ConfiguracaoEixo) -> float:
    """Retorna o momento polar de inércia da seção [m^4] na posição axial x [m]."""
    d = calcular_diametro(x, cfg)
    return np.pi * d**4 / 32.0


# ==========================================================
# 4. INTEGRAÇÃO NUMÉRICA (GAUSS-LEGENDRE POR TRECHOS)
# ==========================================================

def gauss_legendre(
    func: Callable[[float], float],
    inicio: float,
    fim: float,
    n: int = 8,
) -> float:
    """Integra `func` no intervalo [inicio, fim] via quadratura de Gauss-Legendre."""
    if fim <= inicio:
        return 0.0

    nos, pesos = leggauss(n)
    meio = 0.5 * (inicio + fim)
    metade = 0.5 * (fim - inicio)

    soma = sum(wi * func(meio + metade * xi) for xi, wi in zip(nos, pesos))
    return metade * soma


def integrar_trechos(
    func: Callable[[float], float],
    inicio: float,
    fim: float,
    pontos_quebra: list[float],
    n: int = 8,
) -> float:
    """Subdivide o intervalo nos pontos de descontinuidade e integra trecho a trecho."""
    if fim <= inicio:
        return 0.0

    limites = [inicio] + [p for p in pontos_quebra if inicio < p < fim] + [fim]
    return sum(
        gauss_legendre(func, a, b, n)
        for a, b in zip(limites[:-1], limites[1:])
    )


# ==========================================================
# 5. SOLUÇÃO ESTRUTURAL E DINÂMICA (FLEXÃO, TORÇÃO E VIBRAÇÃO)
# ==========================================================

def resolver_plano(forcas: np.ndarray, cfg: ConfiguracaoEixo) -> dict:
    """Calcula reações, esforços internos, rotação e deflexão em um plano de flexão."""
    RB = -np.sum(forcas * cfg.posicoes) / cfg.L
    RA = -np.sum(forcas) - RB

    def cortante(x):
        x_arr = np.asarray(x)
        V = np.full_like(x_arr, RA, dtype=float)
        for F, p in zip(forcas, cfg.posicoes):
            V = V + np.where(x_arr >= p, F, 0.0)
        return float(V) if x_arr.ndim == 0 else V

    def momento(x):
        x_arr = np.asarray(x)
        M = RA * x_arr
        for F, p in zip(forcas, cfg.posicoes):
            M = M + np.where(x_arr >= p, F * (x_arr - p), 0.0)
        return float(M) if x_arr.ndim == 0 else M

    def curvatura(x: float) -> float:
        return momento(x) / (cfg.E * calcular_inercia(x, cfg))

    theta0 = -integrar_trechos(
        lambda s: (cfg.L - s) * curvatura(s),
        0.0,
        cfg.L,
        cfg.pontos_descontinuidade,
        cfg.n_gauss,
    ) / cfg.L

    def rotacao(x: float) -> float:
        return theta0 + integrar_trechos(
            curvatura, 0.0, x, cfg.pontos_descontinuidade, cfg.n_gauss
        )

    def deflexao(x: float) -> float:
        return theta0 * x + integrar_trechos(
            lambda s: (x - s) * curvatura(s),
            0.0,
            x,
            cfg.pontos_descontinuidade,
            cfg.n_gauss,
        )

    return {
        "RA": RA,
        "RB": RB,
        "cortante": cortante,
        "momento": momento,
        "curvatura": curvatura,
        "rotacao": rotacao,
        "deflexao": deflexao,
    }


def calcular_torque(x, cfg: ConfiguracaoEixo):
    """Retorna o diagrama de momento torçor T(x) [N·m] ao longo do eixo."""
    x_arr = np.asarray(x)
    p_pinhao, p_coroa = cfg.posicoes[0], cfg.posicoes[1]
    T = np.where((x_arr >= p_pinhao) & (x_arr <= p_coroa), cfg.T_transmissao, 0.0)
    return float(T) if x_arr.ndim == 0 else T


def avaliar_deflexao_cargas(
    resultado_Y: dict,
    resultado_Z: dict,
    cfg: ConfiguracaoEixo,
) -> list[dict]:
    """Calcula a deflexão exata (dy, dz e resultante dR) nos pontos de aplicação das forças."""
    registros = []
    for nome, xp in zip(cfg.nomes_cargas, cfg.posicoes):
        dy_mm = resultado_Y["deflexao"](float(xp)) * 1000.0
        dz_mm = resultado_Z["deflexao"](float(xp)) * 1000.0
        dR_mm = float(np.hypot(dy_mm, dz_mm))
        registros.append({
            "nome": nome,
            "x_mm": float(xp) * 1000.0,
            "dy_mm": dy_mm,
            "dz_mm": dz_mm,
            "dR_mm": dR_mm,
        })
    return registros


def calcular_velocidade_critica(
    deflexoes_cargas: list[dict],
    cfg: ConfiguracaoEixo,
) -> dict:
    """Calcula a frequência natural de flexão e a velocidade crítica (Método de Rayleigh)."""
    deltas_m = np.array([item["dR_mm"] / 1000.0 for item in deflexoes_cargas], dtype=float)

    numerador = cfg.g * np.sum(cfg.massas * deltas_m)
    denominador = np.sum(cfg.massas * deltas_m**2)

    omega_rad_s = float(np.sqrt(numerador / denominador))
    freq_hz = omega_rad_s / (2.0 * np.pi)
    rpm_critica = freq_hz * 60.0

    return {
        "omega_rad_s": omega_rad_s,
        "freq_hz": freq_hz,
        "rpm_critica": rpm_critica,
        "deltas_m": deltas_m,
    }


def calcular_frequencia_torcional(cfg: ConfiguracaoEixo) -> dict:
    """Calcula a frequência natural torcional entre pinhão e coroa (Shigley, Eq. 10.28)."""
    p_pinhao, p_coroa = float(cfg.posicoes[0]), float(cfg.posicoes[1])
    x_ini, x_fim = min(p_pinhao, p_coroa), max(p_pinhao, p_coroa)
    l = x_fim - x_ini

    # Momentos de inércia de massa dos discos: I = 1/2 * m * r²
    I_discos = 0.5 * cfg.massas * (cfg.raios**2)
    I1, I2 = float(I_discos[0]), float(I_discos[1])

    # Flexibilidade e rigidez torcional equivalente no trecho escalonado/cônico
    flexibilidade = integrar_trechos(
        lambda x: 1.0 / (cfg.G * calcular_momento_polar(x, cfg)),
        x_ini,
        x_fim,
        cfg.pontos_descontinuidade,
        cfg.n_gauss,
    )
    kt = 1.0 / flexibilidade
    J_eq = (kt * l) / cfg.G
    d_eq = ((32.0 * J_eq) / np.pi) ** 0.25

    # Localização do nó (Eq. 10.28b) e rigidezes parciais
    l1 = (I2 * l) / (I1 + I2)
    l2 = l - l1
    k1 = (J_eq * cfg.G) / l1
    k2 = (J_eq * cfg.G) / l2

    # Frequência natural torcional (Eq. 10.28c)
    wn = float(np.sqrt(kt * (I1 + I2) / (I1 * I2)))
    fn = wn / (2.0 * np.pi)
    rpm_torcional = fn * 60.0
    razao_theta = -l1 / l2

    return {
        "l": l,
        "d_eq": d_eq,
        "J_eq": J_eq,
        "I1": I1,
        "I2": I2,
        "kt": kt,
        "l1": l1,
        "l2": l2,
        "k1": k1,
        "k2": k2,
        "wn": wn,
        "fn": fn,
        "rpm_torcional": rpm_torcional,
        "razao_theta": razao_theta,
    }


# ==========================================================
# 6. RELATÓRIO E VISUALIZAÇÃO GRÁFICA
# ==========================================================

def imprimir_resumo(
    resultado_Y: dict,
    resultado_Z: dict,
    x_mm: np.ndarray,
    M_R: np.ndarray,
    T_vals: np.ndarray,
    deflexoes_cargas: list[dict],
    delta_R_mm: np.ndarray | None = None,
    rayleigh: dict | None = None,
    torcional: dict | None = None,
    cfg: ConfiguracaoEixo | None = None,
) -> None:
    """Imprime um resumo numérico completo no terminal."""
    idx_M_max = int(np.argmax(M_R))

    print("=" * 68)
    print("RESUMO DA ANÁLISE ESTRUTURAL DO EIXO")
    print("=" * 68)
    print(f"Reações Plano XY : RA_y = {resultado_Y['RA']:9.2f} N | RB_y = {resultado_Y['RB']:9.2f} N")
    print(f"Reações Plano XZ : RA_z = {resultado_Z['RA']:9.2f} N | RB_z = {resultado_Z['RB']:9.2f} N")
    print("-" * 68)
    print(f"Momento Máx (M_R): {M_R[idx_M_max]:9.2f} N·m em x = {x_mm[idx_M_max]:.2f} mm")
    print(f"Torque Máximo    : {np.max(np.abs(T_vals)):9.2f} N·m")

    if delta_R_mm is not None:
        idx_d_max = int(np.argmax(delta_R_mm))
        print(f"Deflexão Máxima  : {delta_R_mm[idx_d_max]:9.5f} mm em x = {x_mm[idx_d_max]:.2f} mm")

    print("-" * 68)
    print("DEFLEXÃO NOS PONTOS DE ATUAÇÃO DAS FORÇAS:")
    for item in deflexoes_cargas:
        print(
            f"  • {item['nome']:<8} (x = {item['x_mm']:6.2f} mm): "
            f"δ_y = {item['dy_mm']:+.5f} mm | "
            f"δ_z = {item['dz_mm']:+.5f} mm | "
            f"δ_R = {item['dR_mm']:.5f} mm"
        )

    if rayleigh is not None and cfg is not None:
        print("-" * 68)
        print("FREQUÊNCIA NATURAL DE FLEXÃO - MÉTODO DE RAYLEIGH:")
        for nome, m in zip(cfg.nomes_cargas, cfg.massas):
            print(f"  • Massa {nome:<8}: {m:.2f} kg")
        print(f"  • ω_n (angular)  : {rayleigh['omega_rad_s']:.2f} rad/s")
        print(f"  • f_n (natural)  : {rayleigh['freq_hz']:.2f} Hz")
        print(f"  • Veloc. Crítica : {rayleigh['rpm_critica']:.2f} rpm")

    if torcional is not None:
        print("-" * 68)
        print("FREQUÊNCIA NATURAL TORCIONAL (PINHÃO - COROA):")
        print(f"  • Distância entre discos (l) : {torcional['l'] * 1000:.2f} mm")
        print(f"  • Diâmetro equiv. (d_eq)     : {torcional['d_eq'] * 1000:.2f} mm (J_eq = {torcional['J_eq']:.4e} m^4)")
        print(f"  • Inércia Pinhão (I1)        : {torcional['I1']:.6f} kg·m²")
        print(f"  • Inércia Coroa  (I2)        : {torcional['I2']:.6f} kg·m²")
        print(f"  • Rigidez torcional (kt)     : {torcional['kt']:.2f} N·m/rad")
        print(f"  • Posição do nó (l1 | l2)    : {torcional['l1'] * 1000:.2f} mm | {torcional['l2'] * 1000:.2f} mm")
        print(f"  • Rigidezes (k1 | k2)        : {torcional['k1']:.2f} | {torcional['k2']:.2f} N·m/rad")
        print(f"  • ω_nt (angular)             : {torcional['wn']:.2f} rad/s")
        print(f"  • f_nt (natural)             : {torcional['fn']:.2f} Hz ({torcional['rpm_torcional']:.2f} rpm)")
        print(f"  • Razão angular (θ1 / θ2)    : {torcional['razao_theta']:.4f}")

    print("=" * 68)


def _finalizar_figura(fig: plt.Figure, nome_arquivo: str, painel: PainelControle) -> None:
    """Controla o salvamento e a exibição da figura conforme o painel de controle."""
    plt.tight_layout()
    if painel.salvar_figuras:
        destino = Path(painel.pasta_saida)
        destino.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino / nome_arquivo, dpi=300, bbox_inches="tight")

    if painel.plot:
        plt.show()
    else:
        plt.close(fig)


def plotar_momentos_planos(
    x_mm: np.ndarray,
    M_y: np.ndarray,
    M_z: np.ndarray,
    painel: PainelControle,
) -> None:
    """Gera os gráficos de momento fletor nos planos XY e XZ."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(x_mm, M_y, color="tab:red", linewidth=2, label="$M_{xy}(x)$")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Momento Fletor - Plano XY ($M_{xy}$)")
    ax1.set_ylabel("Momento [N·m]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(x_mm, M_z, color="tab:purple", linewidth=2, label="$M_{xz}(x)$")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title("Momento Fletor - Plano XZ ($M_{xz}$)")
    ax2.set_xlabel("Posição x [mm]")
    ax2.set_ylabel("Momento [N·m]")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    _finalizar_figura(fig, "momentos_planos.png", painel)


def plotar_resultante_e_torque(
    x_mm: np.ndarray,
    M_R: np.ndarray,
    T_vals: np.ndarray,
    painel: PainelControle,
) -> None:
    """Gera os gráficos de momento fletor resultante e torque."""
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax3.plot(x_mm, M_R, color="darkred", linewidth=2, label="Momento Resultante $M_R$")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.set_title("Momento Fletor Resultante $M_R(x) = \\sqrt{M_{xy}^2 + M_{xz}^2}$")
    ax3.set_ylabel("Momento [N·m]")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    ax4.plot(x_mm, T_vals, color="darkorange", linewidth=2, label="Torque $T(x)$")
    ax4.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax4.set_title("Diagrama de Momento Torçor / Torque $T(x)$")
    ax4.set_xlabel("Posição x [mm]")
    ax4.set_ylabel("Torque [N·m]")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend()

    _finalizar_figura(fig, "resultante_e_torque.png", painel)


def plotar_deflexao(
    x_mm: np.ndarray,
    delta_R_mm: np.ndarray,
    deflexoes_cargas: list[dict],
    painel: PainelControle,
) -> None:
    """Gera o gráfico da deflexão resultante destacando os pontos de aplicação das forças."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(x_mm, delta_R_mm, color="tab:blue", linewidth=2, label="Deflexão Resultante $\\delta_R(x)$")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    xs_carga = [item["x_mm"] for item in deflexoes_cargas]
    ys_carga = [item["dR_mm"] for item in deflexoes_cargas]
    ax.scatter(xs_carga, ys_carga, color="crimson", zorder=5, s=55, label="Pontos de Carga")

    for item in deflexoes_cargas:
        ax.axvline(item["x_mm"], color="crimson", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"{item['nome']} ({item['x_mm']:.2f} mm)\nδ_R = {item['dR_mm']:.5f} mm",
            xy=(item["x_mm"], item["dR_mm"]),
            xytext=(8, -22),
            textcoords="offset points",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.85),
        )

    ax.set_title("Deflexão Resultante $\\delta_R(x) = \\sqrt{\\delta_y^2 + \\delta_z^2}$")
    ax.set_xlabel("Posição x [mm]")
    ax.set_ylabel("Deflexão [mm]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    _finalizar_figura(fig, "deflexao_resultante.png", painel)


# ==========================================================
# 7. PONTO DE ENTRADA (EXECUÇÃO PRINCIPAL)
# ==========================================================

def main(
    cfg: ConfiguracaoEixo | None = None,
    painel: PainelControle | None = None,
) -> dict:
    cfg = cfg or ConfiguracaoEixo()
    painel = painel or PainelControle()

    # Resolução dos planos de flexão
    resultado_Y = resolver_plano(cfg.Fy, cfg)
    resultado_Z = resolver_plano(cfg.Fz, cfg)

    # Avaliação exata da deflexão nos pontos de aplicação das forças
    deflexoes_cargas = avaliar_deflexao_cargas(resultado_Y, resultado_Z, cfg)

    # Frequências naturais (Flexão via Rayleigh e Torcional entre engrenagens)
    rayleigh = calcular_velocidade_critica(deflexoes_cargas, cfg) if painel.calcular_rayleigh else None
    torcional = calcular_frequencia_torcional(cfg) if painel.calcular_torcional else None

    # Discretização do eixo para diagramas contínuos
    x_vals = np.linspace(0.0, cfg.L, cfg.n_pontos)
    x_mm = x_vals * 1000.0

    M_y = resultado_Y["momento"](x_vals)
    M_z = resultado_Z["momento"](x_vals)
    M_R = np.hypot(M_y, M_z)
    T_vals = calcular_torque(x_vals, cfg)

    delta_R_mm = None
    if painel.plot_deflexao or painel.imprimir_resumo:
        dy = np.array([resultado_Y["deflexao"](x) for x in x_vals])
        dz = np.array([resultado_Z["deflexao"](x) for x in x_vals])
        delta_R_mm = np.hypot(dy, dz) * 1000.0

    if painel.imprimir_resumo:
        imprimir_resumo(
            resultado_Y,
            resultado_Z,
            x_mm,
            M_R,
            T_vals,
            deflexoes_cargas,
            delta_R_mm,
            rayleigh,
            torcional,
            cfg,
        )

    if painel.plot or painel.salvar_figuras:
        plt.style.use("seaborn-v0_8-whitegrid")

        if painel.plot_planos:
            plotar_momentos_planos(x_mm, M_y, M_z, painel)

        if painel.plot_resultante_torque:
            plotar_resultante_e_torque(x_mm, M_R, T_vals, painel)

        if painel.plot_deflexao and delta_R_mm is not None:
            plotar_deflexao(x_mm, delta_R_mm, deflexoes_cargas, painel)

    return {
        "x_mm": x_mm,
        "M_y": M_y,
        "M_z": M_z,
        "M_R": M_R,
        "T": T_vals,
        "delta_R_mm": delta_R_mm,
        "deflexoes_cargas": deflexoes_cargas,
        "rayleigh": rayleigh,
        "torcional": torcional,
        "plano_Y": resultado_Y,
        "plano_Z": resultado_Z,
    }


if __name__ == "__main__":
    main()
