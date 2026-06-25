---
slug: md-component-engine-01-compilable-redesign
criteria:
  - id: ac-001
    summary: ForceField/Potential 组件分层(镜像 molpy),无闭包
    type: code
    pass_when: |
      src/molix/md/forcefield.py 定义 ForceField(nn.Module) 抽象
      (forward(pos: Tensor) -> ForceOutput, 便捷 calc_energy/calc_forces),
      PotentialForceField(ForceField) 绑定一个 molpot Potential + 体系模板,
      HarmonicForceField(ForceField) 提供解析参考。force_seam.py 已删除,
      build_force_fn 闭包不再存在。grep 全 src/molix/md 无 `Callable`
      力别名(ForceFn 删除)、无返回闭包的工厂函数。Integrator 持有
      self.force: ForceField 组件而非 Callable。
    status: verified  # last_checked: 2026-06-26
  - id: ac-002
    summary: 热路径静态化 —— 去 getattr/isinstance/None 分支/运行期 gamma 判断
    type: code
    pass_when: |
      LangevinVerletIntegrator.step 无 `if self.gamma > 0` 分支(O 步恒等式
      始终执行,γ=0 时 c2=0 数值不变),noise 永远是真实张量(无 None 分支);
      mass 在 __init__ 归一化为正张量 buffer(无 isinstance(mass,…));
      MDRunner 的 hook 分发为类型化直调(hook.on_train_batch_end(...)),
      无 getattr(hook, name)。`grep -nE "getattr\(|isinstance\(|is None"
      src/molix/md/integrators.py src/molix/md/forcefield.py` 在 step/forward
      热路径上零命中(构造期允许)。
    status: verified  # last_checked: 2026-06-26
  - id: ac-003
    summary: typed MDState/ForceOutput 作为组件契约(tensordict↔typed 平衡)
    type: code
    pass_when: |
      src/molix/md/types.py 定义 ForceOutput(NamedTuple: energy, forces) 与
      MDState(NamedTuple: pos, vel, force);二者是 pytree(tree_flatten/
      tree_unflatten round-trip 测试通过)。TensorDict 只出现在 ForceField
      内部(模型 I/O),不出现在 Integrator/MDRunner/Hook 的跨组件签名里;
      Integrator.step 的签名是 (MDState, Tensor) -> MDState。
    status: verified  # last_checked: 2026-06-26
  - id: ac-004
    summary: Integrator.rollout 全图编译含 PiNet 力,0 graph-break,compile==eager
    type: code
    pass_when: |
      tests/test_molix/test_md_compile.py 用真实 tiny-PiNet 构造
      PotentialForceField,torch.compile(ig.rollout, fullgraph=True) 跑 N>=5 步
      不抛 graph-break(fullgraph=True 下 break 即抛错→测试失败),且与 eager
      rollout 数值一致(fp64 allclose atol<=1e-10);torch.compile(ig.step) 同样
      0 break 且 ==eager。NVE(γ=0)与 Langevin(γ>0)两路都覆盖。
    status: verified  # last_checked: 2026-06-26
  - id: ac-005
    summary: 编译路径 fp32 与 fp64 均正确
    type: scientific
    pass_when: |
      同一编译验收在 torch.float32 与 torch.float64 下都通过:fp64 与 eager
      位级/1e-10 一致;fp32 与 eager 在归约精度噪声内(atol 适配,记录实测
      gap)。证明可编译 MD 步对两种精度都数值可信。
    status: pending
  - id: ac-006
    summary: 正确性回归全部保留(防重构倒退)
    type: code
    pass_when: |
      迁移后这些断言仍绿:活体几何(test_force_seam_tracks_live_geometry 等价:
      能量/力随非刚性位移变化 >1e-6,冻结-PES 不回归)、energy-varies、
      NVE 能量守恒(drift<1e-3)、Langevin 等分(measured kbt 偏差<5%,dof
      约定 γ>0→3N)、force-caching 位级一致、step==step_cached、
      TrajectoryHook shard-flush==单缓冲且清理 shard、ΔF 在 fp64 相减、
      mass<=0 在构造期 raise ValueError。
    status: verified  # last_checked: 2026-06-26
  - id: ac-007
    summary: MD 套件 + lint/format 全绿,无外部破坏
    type: runtime
    pass_when: |
      PYTHONPATH=src:. <hpc-python> -m pytest tests/test_molix/test_md_integrators.py
      tests/test_molix/test_md_dynamics.py tests/test_molix/test_md_runner.py
      tests/test_molix/test_md_compile.py 全绿;ruff check src/molix/md 与
      ruff format --check src/molix/md 干净;无 src 内其它模块导入已删除的
      force_seam/ForceFn(grep 确认)。__init__.py 导出新组件词表。
    status: verified  # last_checked: 2026-06-26
  - id: ac-008
    summary: LJ 粒子体系 NVE 5 ns 能量不漂移(长程积分稳定性)
    type: scientific
    pass_when: |
      新增 LennardJonesForceField(ForceField)(全对 LJ E=4ε[(σ/r)^12−(σ/r)^6],
      无 cutoff/无邻居表→全对天然规避冻结邻居表问题,解析可微)与验证脚本
      benchmarks/verify_md_lj_nve.py:对一个 LJ 团簇(脚本记录 N(~13–55)、
      σ/ε/m、dt、初始温度 T0、平衡构型来源)以 γ=0(NVE)经 torch.compile 的
      rollout 跑满 5 ns(步数 = 5ns/dt,数百万步,只有编译/CUDA-graph 才可行)。
      判据:总能量无系统漂移 —— 对 E_tot(t) 线性拟合,
      |slope·(5 ns)| / |E_tot(0)| < 1e-3,且能量 RMS 涨落有界(不发散、无单调
      爬升),温度保持有限不爆炸。脚本打印 drift、拟合斜率与 PASS/FAIL;
      同时记录 steps/s 作为编译性能旁证。
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003 是"组件化 + 静态化"三条硬指令的可证伪化**:分别钉死(1)ForceField≠Potential 的 molpy 式分层且无闭包、(2)热路径无动态判断、(3)typed state 作为组件契约。
- **ac-004 / ac-005 是"`torch.compile(md.run)` 全流程编译"的核心验收**:`rollout`(含 PiNet functorch 力)fullgraph 0 break 且 compile==eager,fp32/fp64 双精度。这是本次重构区别于上一版的根本能力。
- **ac-006 是防倒退闸**:上一轮 review 修掉的 🚨 冻结-PES 与全部 🔴(显存/ΔF fp64/dof/单位)必须在组件化后逐条仍然成立。
- **ac-008 是 NVE 积分器金标准**:LJ 团簇(刚硬非谐真实 PES)NVE 5 ns 总能量无系统漂移 —— 比谐振子玩具强得多的长程稳定性检验,且 5ns≈数百万步只有编译路径可行,顺带压测性能。
- 全部 8 条 verified 后方可 code-complete。
