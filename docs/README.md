# 足球数据与预测平台文档

## 当前权威文档

- [总体设计](superpowers/specs/2026-07-16-football-data-platform-design.md)
- [领域术语表](domain-glossary.md)

## 架构决策记录

- [ADR-0001：数据平台为核心，价值投注为下游模块](adr/0001-data-platform-and-betting-boundary.md)
- [ADR-0002：第一阶段核心赛事为欧洲五大联赛](adr/0002-core-competition-coverage.md)
- [ADR-0003：首要比赛预测目标为赛前 90 分钟联合比分分布](adr/0003-primary-match-prediction-target.md)
- [ADR-0004：零预算数据源策略](adr/0004-zero-budget-data-source-strategy.md)
- [ADR-0005：新闻证据与结构化事件](adr/0005-news-evidence-and-structured-events.md)
- [ADR-0006：球员采用位置与角色画像](adr/0006-role-based-player-profiles.md)
- [ADR-0007：球队基线、阵容增量与上下文](adr/0007-team-baseline-lineup-delta-and-context.md)
- [ADR-0008：基准与挑战者模型治理](adr/0008-champion-challenger-model-governance.md)
- [ADR-0009：核心足球模型与市场赔率分离](adr/0009-separate-football-model-from-market-odds.md)
- [ADR-0010：第一阶段不做滚球](adr/0010-no-in-play-prediction-or-betting-in-phase-one.md)
- [ADR-0011：赛事模拟作为可选下游应用](adr/0011-tournament-simulation-as-optional-downstream-application.md)
- [ADR-0012：本地命令行流水线与静态报告](adr/0012-local-cli-pipelines-and-static-reports.md)
- [ADR-0013：版本化比赛数据生命周期](adr/0013-versioned-match-data-lifecycle.md)
- [ADR-0014：按任务认证训练准入](adr/0014-task-specific-training-readiness.md)
- [ADR-0015：平台实体 ID 与来源映射](adr/0015-canonical-entity-identities-and-source-mappings.md)
- [ADR-0016：raw、canonical、derived 三层数据](adr/0016-raw-canonical-derived-data-layers.md)
- [ADR-0017：先完成 2025-26 英超纵向切片](adr/0017-premier-league-2025-26-vertical-slice-first.md)
- [ADR-0018：区分实采与历史重建快照](adr/0018-distinguish-captured-and-reconstructed-snapshots.md)
- [ADR-0019：第一阶段价值投注仅限纸面分析](adr/0019-paper-only-value-betting-in-phase-one.md)
- [ADR-0020：产品命名与仓库兼容](adr/0020-product-name-and-repository-compatibility.md)
- [ADR-0021：新实现进入 football-data-platform 仓库](adr/0021-new-implementation-repository.md)
- [ADR-0022：Football-Data 仅作为赛程与赛果备用源](adr/0022-football-data-results-fallback.md)

## 实施与验收

- [当前交付计划与自动化验收门禁](../PROJECT_PLAN.md)

## 历史设计

- [2026-05-28 世界杯预测系统设计](superpowers/specs/2026-05-28-world-cup-prediction-design.md) — 已被替代，仅作历史参考。

后续新增架构决策应以 ADR 记录，并同步更新总体设计或术语表中受影响的当前口径。
