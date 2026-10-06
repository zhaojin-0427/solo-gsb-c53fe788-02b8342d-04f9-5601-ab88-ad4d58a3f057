"""数值正确性单元测试：抽样分布、GUM 解析解、MC 大数收敛。"""

from __future__ import annotations

import math

import numpy as np

from app.engine import _marginal_unit_samples, _phi, _norm_ppf


def test_phi_matches_known_values():
    assert abs(float(_phi(np.array([0.0]))[0]) - 0.5) < 1e-7
    assert abs(float(_phi(np.array([1.96]))[0]) - 0.975) < 1e-5
    assert abs(float(_phi(np.array([-1.96]))[0]) - 0.025) < 1e-5
    assert abs(float(_phi(np.array([3.0]))[0]) - 0.99865) < 1e-5


def test_norm_ppf_inverts_phi():
    p = np.linspace(0.001, 0.999, 999)
    z = _norm_ppf(p)
    back = _phi(z)
    assert np.max(np.abs(back - p)) < 3e-6
    assert abs(float(z[np.argmin(np.abs(p - 0.975))]) - 1.96) < 1e-4


def test_marginal_distributions_have_unit_variance():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=500_000)
    for dist, target_mean in (("normal", 0.0), ("uniform", 0.0), ("triangular", 0.0)):
        s = _marginal_unit_samples(dist, u.copy())
        assert abs(s.mean() - target_mean) < 2e-3
        assert abs(s.std(ddof=1) - 1.0) < 2e-3


def test_triangular_support_bounds():
    u = np.linspace(1e-9, 1 - 1e-9, 10000)
    s = _marginal_unit_samples("triangular", u)
    assert s.min() >= -math.sqrt(6.0) - 1e-6
    assert s.max() <= math.sqrt(6.0) + 1e-6


def test_gum_linear_model_exact_variance():
    # y = 2*a + 3*b（无量纲线性），相关 r -> GUM 必须与解析解一致
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_gum

    expr = compile_expression("2*a + 3*b", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    x = ctx.nominal_to_base({"a": 1.0, "b": -2.0})
    std = np.array([0.1, 0.2])
    for r in (-0.7, 0.0, 0.5):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        g = run_gum(expr, ("a", "b"), x, std, corr, ctx)
        expected_var = (2 * 0.1) ** 2 + (3 * 0.2) ** 2 + 2 * 2 * 3 * r * 0.1 * 0.2
        assert abs(g.y - (-4.0)) < 1e-9
        assert abs(g.uc - math.sqrt(expected_var)) < 1e-9


def test_monte_carlo_matches_gum_for_linear_gaussian():
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_monte_carlo

    expr = compile_expression("2*a + 3*b", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    x = {"a": 0.0, "b": 0.0}
    std = np.array([0.1, 0.2])
    corr = build_correlation_matrix(("a", "b"), [("a", "b", 0.5)], set())
    mc = run_monte_carlo(
        expr, ("a", "b"), ("normal", "normal"), x, std, corr,
        n_draws=100_000, seed=123, ctx=ctx, convergence_tol=0.1,
    )
    expected_sd = math.sqrt(0.2 ** 2 + 0.6 ** 2 + 2 * 0.5 * 0.2 * 0.6)
    assert abs(mc.mean) < 3e-3
    assert abs(mc.std - expected_sd) < 3e-3
    lo, hi = -1.96 * expected_sd, 1.96 * expected_sd
    assert abs(mc.coverage_low - lo) < 1.5e-2
    assert abs(mc.coverage_high - hi) < 1.5e-2
    assert mc.converged


def test_reproducibility_fixed_seed():
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_monte_carlo

    expr = compile_expression("a*b + sqrt(a**2)", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    x = {"a": 2.0, "b": 3.0}
    std = np.array([0.05, 0.05])
    corr = np.eye(2)
    kw = dict(compiled=expr, names=("a", "b"), distributions=("normal", "normal"),
              x_base=x, std_base=std, corr=corr, n_draws=5000, ctx=ctx, convergence_tol=0.5)
    m1 = run_monte_carlo(seed=99, **kw)
    m2 = run_monte_carlo(seed=99, **kw)
    assert m1.mean == m2.mean and m1.coverage_low == m2.coverage_low
    m3 = run_monte_carlo(seed=100, **kw)
    assert m3.mean != m1.mean


def test_psd_repair_accepts_tiny_negative_eigenvalue():
    from app.engine import build_correlation_matrix, CorrelationError
    # 合法相关系数（全部 <= 1）组成、但整体含 -5e-10 量级负特征值的近奇异矩阵
    pairs = [
        ("a", "b", 1.0 - 1e-11),
        ("a", "c", 1.0 - 1e-11),
        ("b", "c", 0.9999999995),
    ]
    m = np.eye(3)
    m[0, 1] = m[1, 0] = pairs[0][2]
    m[0, 2] = m[2, 0] = pairs[1][2]
    m[1, 2] = m[2, 1] = pairs[2][2]
    assert -1e-8 < np.linalg.eigvalsh(m)[0] < 0
    r = build_correlation_matrix(("a", "b", "c"), pairs, set())
    assert r.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(r) >= -1e-12)
    # 修复必须保持单位对角
    assert np.allclose(np.diag(r), 1.0)
    # 明显非 PSD 拒绝
    try:
        build_correlation_matrix(("a", "b", "c"),
                                 [("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", -0.9)], set())
        assert False, "应当拒绝非半正定矩阵"
    except CorrelationError:
        pass


def test_perfect_correlation_singular_matrix_accepted():
    # 两个非零标准不确定度输入、r 恰为 ±1：矩阵奇异（零特征值）但合法，必须接受
    from app.engine import build_correlation_matrix
    for r in (1.0, -1.0):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        evals = np.linalg.eigvalsh(corr)
        assert abs(evals[0]) < 1e-12
        assert abs(evals[1] - 2.0) < 1e-12
        # Cholesky 在 r=-1 上必然失败，但矩阵不得因此被拒
        np.testing.assert_allclose(corr, [[1.0, r], [r, 1.0]])


def test_correlation_factor_reproduces_matrix():
    from app.engine import build_correlation_matrix, correlation_factor
    for r in (1.0, -1.0, 0.3):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        f = correlation_factor(corr)
        np.testing.assert_allclose(f @ f.T, corr, atol=1e-12)
    # 正定矩阵仍走 Cholesky（下三角）
    f = correlation_factor(build_correlation_matrix(("a", "b"), [("a", "b", 0.5)], set()))
    assert abs(float(f[0, 1])) < 1e-15


def test_reject_threshold_is_one_e_minus_eight():
    # 最小特征值 < -1e-8 必须拒绝；[-1e-8, 0) 的舍入误差修复放行（保单位对角）
    from app.engine import build_correlation_matrix, CorrelationError

    def mat3(r_ab, r_bc, r_ac):
        return build_correlation_matrix(
            ("a", "b", "c"),
            [("a", "b", r_ab), ("b", "c", r_bc), ("a", "c", r_ac)], set(),
        )

    # 矛盾的 ±1 组合：特征值 1±√3，最小 ≈ -0.73
    try:
        mat3(1.0, 1.0, -1.0)
        assert False, "明显非 PSD 必须拒绝"
    except CorrelationError:
        pass

    pairs = [
        ("a", "b", 1.0 - 1e-11),
        ("a", "c", 1.0 - 1e-11),
        ("b", "c", 0.9999999995),
    ]
    raw = np.eye(3)
    raw[0, 1] = raw[1, 0] = pairs[0][2]
    raw[0, 2] = raw[2, 0] = pairs[1][2]
    raw[1, 2] = raw[2, 1] = pairs[2][2]
    assert -1e-8 <= np.linalg.eigvalsh(raw)[0] < 0
    fixed = mat3(*[p[2] for p in pairs])
    assert np.min(np.linalg.eigvalsh(fixed)) >= -1e-12
    np.testing.assert_allclose(np.diag(fixed), 1.0)


def _linear_sum_ctx():
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    expr = compile_expression("a + b", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    return expr, ctx


def test_gum_perfect_correlation_uncertainty_is_abs_sum():
    # y = a + b、正态输入：GUM 标准不确定度必须为 |ua + r·ub|
    from app.engine import build_correlation_matrix, run_gum
    expr, ctx = _linear_sum_ctx()
    x = ctx.nominal_to_base({"a": 3.0, "b": 5.0})
    ua, ub = 0.1, 0.2
    std = np.array([ua, ub])
    for r, expected in ((1.0, ua + ub), (-1.0, abs(ua - ub))):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        g = run_gum(expr, ("a", "b"), x, std, corr, ctx)
        assert abs(g.y - 8.0) < 1e-9
        assert abs(g.uc - expected) < 1e-9, (r, g.uc, expected)

    # ua == ub、r = -1：完全抵消，uc = 0，覆盖区间退化为点
    corr = build_correlation_matrix(("a", "b"), [("a", "b", -1.0)], set())
    g = run_gum(expr, ("a", "b"), x, np.array([0.1, 0.1]), corr, ctx)
    assert g.uc == 0.0
    assert g.coverage_low == g.coverage_high == 8.0


def test_monte_carlo_perfect_correlation_directions():
    # 固定种子下：r=+1 标准化输入同向（ua+ub），r=-1 反向（|ua-ub|）
    from app.engine import (
        build_correlation_matrix, correlation_factor, run_monte_carlo,
    )
    expr, ctx = _linear_sum_ctx()
    x = {"a": 0.0, "b": 0.0}
    ua, ub = 0.1, 0.2
    std = np.array([ua, ub])

    for r, expected_sd in ((1.0, ua + ub), (-1.0, abs(ua - ub))):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        mc = run_monte_carlo(
            expr, ("a", "b"), ("normal", "normal"), x, std, corr,
            n_draws=20_000, seed=20240101, ctx=ctx, convergence_tol=0.05,
        )
        assert mc.converged, mc.nonconvergence_reasons
        assert abs(mc.std - expected_sd) < 5e-3, (r, mc.std, expected_sd)

    # 因子 + 对齐后的逐样本方向关系（直接检查标准化正态）
    rng = np.random.default_rng(7)
    w = rng.standard_normal(size=(2, 1000))
    from app.engine import _align_perfectly_correlated
    for r in (1.0, -1.0):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        z = _align_perfectly_correlated(correlation_factor(corr) @ w, corr)
        if r == 1.0:
            np.testing.assert_array_equal(z[0], z[1])
        else:
            np.testing.assert_array_equal(z[0], -z[1])


def test_gum_tiny_magnitude_variance_not_clamped_to_zero():
    # 回归：旧实现用固定绝对阈值（eps*max(1.0, 对角项)）钳方差，
    # 把 u=1e-9 时真实的 ~1e-18 量级正方差误判为舍入噪声而归零。
    # 阈值必须与协方差项尺度同步收缩，保留真实小正方差。
    from app.engine import build_correlation_matrix, run_gum
    expr, ctx = _linear_sum_ctx()
    x = ctx.nominal_to_base({"a": 3.0, "b": 5.0})

    # r=+1, ua=ub=1e-9 -> uc = 2e-9（不得归零）
    std = np.array([1e-9, 1e-9])
    corr = build_correlation_matrix(("a", "b"), [("a", "b", 1.0)], set())
    g = run_gum(expr, ("a", "b"), x, std, corr, ctx)
    assert abs(g.uc - 2e-9) < 1e-15
    assert g.coverage_low < 8.0 < g.coverage_high
    by_source = {c["input"]: c for c in g.contributions}
    assert by_source["a"]["variance_contribution"] > 0.0
    assert by_source["b"]["variance_contribution"] > 0.0
    assert by_source["a~b"]["variance_contribution"] > 0.0
    assert abs(by_source["a"]["fraction"] - 0.25) < 1e-9
    assert abs(by_source["a~b"]["fraction"] - 0.5) < 1e-9

    # r=-1, ua=1e-9, ub=2e-9 -> uc = 1e-9（不得归零）
    std = np.array([1e-9, 2e-9])
    corr = build_correlation_matrix(("a", "b"), [("a", "b", -1.0)], set())
    g = run_gum(expr, ("a", "b"), x, std, corr, ctx)
    assert abs(g.uc - 1e-9) < 1e-12
    assert g.coverage_low < 8.0 < g.coverage_high
    assert g.contributions  # 逐项贡献仍然给出

    # r=-1, ua=ub=1e-9 -> 完全抵消，uc = 0（尺度收缩后阈值仍能识别真实抵消）
    std = np.array([1e-9, 1e-9])
    corr = build_correlation_matrix(("a", "b"), [("a", "b", -1.0)], set())
    g = run_gum(expr, ("a", "b"), x, std, corr, ctx)
    assert g.uc == 0.0
    assert g.coverage_low == g.coverage_high == 8.0


def test_monte_carlo_tiny_magnitude_converges_with_nonzero_sd():
    # 小量级蒙特卡洛：相对标准误与尺度无关，必须正常收敛且标准差不为零
    from app.engine import build_correlation_matrix, run_monte_carlo
    expr, ctx = _linear_sum_ctx()
    x = {"a": 3.0, "b": 5.0}
    std = np.array([1e-9, 1e-9])
    corr = build_correlation_matrix(("a", "b"), [("a", "b", 1.0)], set())
    mc = run_monte_carlo(
        expr, ("a", "b"), ("normal", "normal"), x, std, corr,
        n_draws=20_000, seed=20261003, ctx=ctx, convergence_tol=0.05,
    )
    assert mc.converged, mc.nonconvergence_reasons
    assert abs(mc.std - 2e-9) < 5e-11
    assert mc.coverage_low < 8.0 < mc.coverage_high


def test_three_way_perfect_correlation_singular_block():
    # 三个完全相关变量（r 全为 1）：秩 1 的奇异矩阵，也必须能登记与抽样
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_monte_carlo
    names = ("a", "b", "c")
    pairs = [("a", "b", 1.0), ("a", "c", 1.0), ("b", "c", 1.0)]
    corr = build_correlation_matrix(names, pairs, set())
    evals = np.linalg.eigvalsh(corr)
    np.testing.assert_allclose(sorted(evals), [0.0, 0.0, 3.0], atol=1e-10)

    expr = compile_expression("a + b + c", names)
    units = {k: parse_unit("dimensionless") for k in names}
    ctx = build_unit_context(expr, units, "dimensionless")
    std = np.array([0.1, 0.2, 0.3])
    mc = run_monte_carlo(
        expr, names, ("normal",) * 3, {k: 0.0 for k in names}, std, corr,
        n_draws=5000, seed=11, ctx=ctx, convergence_tol=0.05,
    )
    assert mc.converged, mc.nonconvergence_reasons
    # 完全同向：y 的标准差 = ua + ub + uc
    assert abs(mc.std - 0.6) < 1e-2


def test_gradient_adaptive_step_recovers_biased_small_slope():
    # 输出偏置淹没小真实斜率：y = 1 + 1e-4*b，b=0。固定极小步长 eps^(2/3)
    # 上差分真值只有偏置 ulp 的 ~33 倍，落在百倍噪声门槛内；自适应放大到
    # eps^(1/3) 尺度后梯度可辨，且精度远高于旧步长（旧梯度误差 ~1.4%）。
    from app.expressions import compile_expression
    from app.engine import (
        GRADIENT_RESOLUTION_FACTOR, numerical_gradient_with_noise,
    )

    expr = compile_expression("a + 1e-4*b", ("a", "b"))
    x = {"a": 1.0, "b": 0.0}
    grad, noise, resolved = numerical_gradient_with_noise(expr, x, ("a", "b"))
    assert abs(grad[0] - 1.0) < 1e-9
    assert abs(grad[1] - 1e-4) < 1e-9
    # 恢复后梯度必须显著高于自身舍入尺度（可辨）
    assert abs(grad[1]) > GRADIENT_RESOLUTION_FACTOR * noise[1]
    assert bool(resolved[1]) is True


def test_gradient_true_flat_direction_is_zero_at_any_scale():
    # 真零方向（2*(a-a)+b 中的 a）逐级放大仍不可辨：梯度严格为 0。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    expr = compile_expression("2*(a - a) + b", ("a", "b"))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"a": 1.0, "b": 2.0}, ("a", "b"))
    assert grad[0] == 0.0
    assert bool(resolved[0]) is False  # 真零方向不可辨
    assert abs(grad[1] - 1.0) < 1e-9


def test_gradient_zero_nominal_without_bias_unchanged():
    # 无偏置零名义点在初始步长即可辨：梯度与零常数项都保持正确。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    expr = compile_expression("a + 1e-13*b", ("a", "b"))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"a": 0.0, "b": 0.0}, ("a", "b"))
    assert grad[0] == 1.0
    assert abs(grad[1] - 1e-13) < 1e-25  # 1e-13 字面量双精度存储尾差 ~1e-29


def test_gradient_widening_stops_at_domain_boundary():
    # 放大步长越过 sqrt 定义域（x-h<0）时保留上一个有限步长，不抛 NaN。
    # y = 1 + 1e-10*sqrt(x)，x0=1e-8：初始差分被偏置 1 淹没触发放大，
    # 放大后的步长越过 0 => NaN 必须被截住；中心差分到不了 h_cap、差分值
    # 达不到偏置舍入门槛，但域内单侧探针确认依赖后该分量判为可辨（非零）。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    expr = compile_expression("1.0 + 1e-10*sqrt(x)", ("x",))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": 1e-8}, ("x",))
    assert np.isfinite(grad[0])
    assert grad[0] > 0.0  # 真实正斜率 1e-10/(2*sqrt(1e-8)) = 5e-7
    assert bool(resolved[0]) is True  # 单侧探针确认依赖：不得误判为零


def test_gradient_domain_boundary_true_zero_stays_zero():
    # 同一边界结构下的真零方向（0.0*sqrt(x)）：单侧探针到 h_cap 差值恒为 0，
    # 维持零结论（梯度严格 0、不可辨）。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    expr = compile_expression("1.0 + 0.0*sqrt(x)", ("x",))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": 1e-8}, ("x",))
    assert grad[0] == 0.0
    assert bool(resolved[0]) is False


# ---------------------------------------------------------------------------
# 首个对称探针越过定义域：名义点有效时域内缩步（asin 右边界镜像 sqrt 左边界）
# ---------------------------------------------------------------------------

def test_gradient_first_probe_beyond_asin_domain_shrinks_inside():
    # y = 1 + asin(x)，x0 = 1-5e-9：初始对称步长 h0 = eps^(1/3) ≈ 6.06e-6
    # 远大于到右边界的距离 5e-9，x0+h0 > 1 使首个探针 NaN。旧实现直接抛
    # PropagationError（错误拒绝有效名义点）；修复后域内缩步取双侧中心
    # 差分，斜率 ≈ 1/sqrt(1-x0²) ≈ 10000（有限、为正、可辨），相对误差
    # 在 1e-5 以内——绝不使用大步单侧一阶差分（其曲率误差 ~94%，斜率
    # 被低估到 ~558）。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    x0 = 1.0 - 5e-9
    expr = compile_expression("1 + asin(x)", ("x",))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": x0}, ("x",))
    exact = 1.0 / math.sqrt(1.0 - x0 * x0)
    assert np.isfinite(grad[0])
    assert grad[0] > 0.0
    assert bool(resolved[0]) is True
    assert abs(grad[0] - exact) / exact < 1e-5
    # 大步单侧一阶差分（错误做法）在该点只有 ~558：修复值必须远高于它
    one_sided_big = (math.asin(x0) - math.asin(x0 - np.finfo(float).eps ** (1.0 / 3.0))) \
        / np.finfo(float).eps ** (1.0 / 3.0)
    assert grad[0] > 10.0 * one_sided_big


def test_gradient_first_probe_beyond_acos_domain_shrinks_inside():
    # acos 在同一右边界镜像场景：x0+h0 > 1。缩步必须在截断/舍入平衡点
    # 停止——若一路缩到 ulp 尺度，相邻浮点表示的舍入会给出偏差 5%~26%
    # 的伪梯度。精确斜率 -1/sqrt(1-x0²) ≈ -10000。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    x0 = 1.0 - 5e-9
    expr = compile_expression("acos(x)", ("x",))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": x0}, ("x",))
    exact = -1.0 / math.sqrt(1.0 - x0 * x0)
    assert bool(resolved[0]) is True
    assert abs(grad[0] - exact) / abs(exact) < 1e-5


def test_gradient_first_probe_beyond_sqrt_domain_tiny_nominal_shrinks():
    # 左边界镜像：sqrt(x) 在 x0 = 1e-12 处 h0 > x0，首个探针 x0-h0 < 0。
    # 域内缩步恢复精确斜率 1/(2 sqrt(x0)) = 5e5。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    expr = compile_expression("sqrt(x)", ("x",))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": 1e-12}, ("x",))
    exact = 0.5 / math.sqrt(1e-12)
    assert bool(resolved[0]) is True
    assert abs(grad[0] - exact) / exact < 1e-8


def test_gradient_first_probe_beyond_domain_true_zero_stays_zero():
    # 首个探针越界缩步场景下的真零斜率（1 + 0.0*asin(x)）：域内中心差分
    # 差值恒为 0、单侧探针也无变化，仍判零（梯度 0、不可辨）。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    x0 = 1.0 - 5e-9
    expr = compile_expression("1.0 + 0.0*asin(x)", ("x",))
    grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": x0}, ("x",))
    assert grad[0] == 0.0
    assert bool(resolved[0]) is False


def test_gradient_nominal_pinned_at_domain_boundary_still_rejected():
    # 名义点恰钉在边界（asin 的 x=±1）：向内一侧最多一个 ulp、向外一侧
    # 必越界，任何浮点可分辨的双侧探针都不存在——GUM 一阶传播无定义，
    # 保持“名义点无效即拒绝”的原语义（PropagationError）。
    import pytest
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise
    from app.errors import PropagationError

    expr = compile_expression("1 + asin(x)", ("x",))
    for pinned in (1.0, -1.0):
        with pytest.raises(PropagationError):
            numerical_gradient_with_noise(expr, {"x": pinned}, ("x",))


def test_gradient_nominal_outside_domain_still_rejected():
    # 名义点本身就在域外（asin 的 x = 1+1e-9）：f(x0) 非有限，直接拒绝，
    # 不进入缩步/探针流程。
    import pytest
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise
    from app.errors import PropagationError

    expr = compile_expression("1 + asin(x)", ("x",))
    with pytest.raises(PropagationError):
        numerical_gradient_with_noise(expr, {"x": 1.0 + 1e-9}, ("x",))


def test_gradient_two_sided_interior_points_unchanged():
    # 两侧都远离边界的内部点：梯度路径与精度不随缩步分支改变。
    from app.expressions import compile_expression
    from app.engine import numerical_gradient_with_noise

    expr = compile_expression("1 + asin(x)", ("x",))
    for x0 in (0.0, 0.5, 0.999, -0.999):
        grad, noise, resolved = numerical_gradient_with_noise(expr, {"x": x0}, ("x",))
        exact = 1.0 / math.sqrt(1.0 - x0 * x0)
        assert bool(resolved[0]) is True
        # 内部点沿用初始 h0 中心差分（截断/舍入平衡尺度），1e-5 级相对精度
        assert abs(grad[0] - exact) / exact < 1e-5
