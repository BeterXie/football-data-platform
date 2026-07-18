# 修复与优化计划

- 状态：执行计划，不是验收证书
- 基线日期：2026-07-16
- 权威依据：[总体设计](superpowers/specs/2026-07-16-football-data-platform-design.md)、
  [ADR-0001 至 ADR-0022](README.md#架构决策记录)、[领域术语表](domain-glossary.md)
- 当前边界：先证明 Premier League 2025-26 纵向切片；两场 golden sample 只证明离线重放

本文把已复现的实现缺口转成可执行工作项和验收证据。状态只能由当前数据、运行清单、
负向测试和端到端产物共同证明；测试通过但未覆盖对应风险时，工作项仍保持未完成。

## Review Checkpoint (2026-07-18，R04 player-fact partial)

本节取代 2026-07-16 checkpoint 中“R01-R12 合同级修复均已完成”和固定测试数量等过强、
已过时的当前状态描述。当前工作树关闭了一批已复现的持久化伪造路径，并完成 R04 的 team/player
fact 局部 checkpoint，但不是十二项全部完成，也不是 Premier League 2025-26 全季或总体设计
第一阶段的验收证书。本 checkpoint 的直接相关完成矩阵为 `393 passed`，且 Ruff、format、compile
和 `git diff --check` 通过；`prediction_evaluation` 与 `governance` 两组各在 1204 秒硬超时前没有
失败，但都未完成，不能计作通过。本批未运行全量 `pytest`，因此不得写成全量回归或统一质量门禁
已经通过。`fdp run-golden --data-root C:\tmp\fdp-r04-player-golden --observed-at
2026-07-16T08:00:00Z` 连续两次均为 exit 0 / status `ok`，run ID 都是
`run:4738458e8ec3b5a079b9ce083ee8f45dc0591d5d285feea1c63a84d85d531764`；
`prediction_id=null` 是既有 fail-closed 预期。该临时目录运行只证明确定性 replay，不是归档证据。

canonical 数据库已通过原子迁移升级到 schema v7。source mapping 现在保存稳定 `mapping_id`、
单调版本、创建者、被替代映射和 UTC 半开有效区间；当前版本唯一、实体类型、版本连续性、区间
不重叠、只允许关闭当前版本及历史不可删除等规则由 SQL 约束和 trigger 强制执行。冲突、修订
事件、人工决定、证据和同组候选作废标记均为 append-only；`revise_source_mapping` 以预期当前
`mapping_id` 执行 CAS，并对完全相同的人工 override 幂等。`resolve_source_mapping` 可按显式版本或
`as_of` 重放历史；未映射或低置信度候选可进入持久化冲突队列。这只是 R02 的第一切片，现有下游
合同尚未精确 pin `mapping_id`/version，也尚未完成跨源生产解析、重资格化和重算闭环。

官方阵容来源使用 JSON schema 2（parser `official-lineup-json/1`）和 canonical official-lineup
contract v2：raw 在解析前归档，来源比赛、球队和 22 个来源球员绑定到平台 ID，双方首发、精确
`fact_evidence` 和 content-addressed contract 在同一事务写入。重放会重新解析 RawArchive 字节并
核对按 `observed_at` 取得的映射、双方 11 人及 22 条 canonical starter facts；legacy contract v1
和 DTO 写入路径均不能重新认证正式阵容。FBref match-report contract 同样会从 raw 重放 parser、
比赛身份、90 分钟赛果和生产表集合，summary-only preview 不能满足完整报告门禁。

本次 R04 partial checkpoint 为 `team_match_observations` 增加 typed `TeamMatchObservation` replay。
loader 会核对内容 ID、精确 `fact_evidence`、RawArchive 注册/manifest/对象字节、`known_at` 与 raw
`target_event_time`、唯一 match-report contract、成功 collection attempt、合同观察时点的比赛/球队
映射，并从同一 raw parser 重算 `aggregated_stats` 和主客 90 分钟进球。availability 同时应用
`known_at <= as_of` 与 `observed_at <= as_of`，每队只评估最新可见候选；最新候选 replay 失败时记录
诊断并 fail closed，不回退旧事实。双方 latest replay 通过后，availability 保留并比较其
match/version/contract/raw/`observed_at`/`known_at` provenance；不一致时保留两条 latest refs，记录
`typed_team_fact_pair_mismatch`，不回退旧组合。lifecycle 会独立重算该配对约束，因而
`archived-complete` 和 `team-baseline-ready` 都要求 typed result ref、配对一致的双方 typed team refs
且无 replay diagnostics；training qualification 会继承计算结果并把实际 result/team refs 固定进产物。
该 checkpoint 还为 match-report player facts 增加不可变 typed `PlayerMatchObservation`、
`ActualLineupFact` 与聚合的 `MatchReportPlayerBatch`。batch loader 从唯一 report contract 重放 raw、
parser 和合同观察时点的历史 match/team/player mapping，逐项核对内容身份、精确 `fact_evidence`、
RawArchive 注册/manifest/对象字节、match/version/team/player、role、minutes、metrics、starter/bench、
`known_at` 与 `observed_at`。availability 同时应用 knowledge/observation 两个 cutoff，只选择最新可见
report batch；最新 batch 无效时保留候选 refs 和诊断且不回退，多个同观察时点但语义不同的 batch
以 ambiguity fail closed。由赛后 match report 产生的 `ActualLineupFact.official=False` 只表示实际
出场名单，不是赛前官方首发，也不能充当 `lineups-confirmed` 或 official-lineup contract 的证据。

完整 22 人离线样本已证明 22 条 typed player observations、22 条 actual-lineup facts、双方各 11 名
starter 能让 `player-profile-ready`、lifecycle 和 formal training qualification 通过，并把精确 batch
contract/fact refs 固定进资格产物。bundled vertical slice 的画像输入不再直接消费 parser DTO：只从
verified canonical player refs 构造，`played_at` 使用 canonical 精确比赛版本的 kickoff，画像 manifest
也只引用这些 canonical refs。`VerificationSession` 已接入 player batch、单条 player observation 和
actual-lineup replay，在同一请求的 DB snapshot/FileProof 上复用成功结果，失败结果不缓存。上述只是
R04 player-fact checkpoint，不代表通用画像 producer、其他来源 lineup/news/event 或其余开放项完成。

正式预测已使用 prediction schema v4 和 score-grid-composition schema v2。持久化预测的共享
校验入口会重新加载并校验 snapshot、snapshot manifest、model run、prediction manifest、完整
calibration policy 和唯一 Dixon-Coles grid；training reference、governance comparison 与 paper
ledger 均复用该入口。贡献产物保存公式版本、来源路径/值、参考值、系数和 bounds，且正式预测
必须恰好消费 policy 规定的 baseline、context 和所有 ready lineup 维度。`legacy-inline/1` 不能
再创建正式预测；prediction schema v3 只能通过独立 audit-only view 读取，并明确保留 provenance、
calibration 和历史训练标签无法重新认证的限制。

正式 `prematch-features/3` 的球队基线只接受 `team-baseline-input/3`。专用 writer 只接收比赛 ID、
精确比赛版本、`as_of` 和已持久化 baseline artifact ID；它从 canonical 精确版本重放主客方向与
开球时间，重新调用 `expected_goals_from_baseline` 计算双方 lambda，并把比赛、版本、主客队、
开球、cutoff、baseline artifact 与 transform 绑定进 content-addressed source context。读取时会
再次精确比较 value、input refs、时间和 context；`team-baseline-input/1`、`/2` 仅保留 audit replay，
通用 source writer 不能新建 `/1`、`/2` 或 `/3`。缺目标球队、未来 baseline cutoff、晚于 cutoff
的输入，以及没有 semantic `known_at`/target event timestamp 的裸 raw 输入均 fail closed。
baseline loader 还要求该 artifact 恰好只有一份 semantic manifest，并把类型、schema、transform、
payload、input/output refs、状态、质量和生成时间重新对照 writer contract；存在第二份认领同一
baseline ID 的清单也会 fail closed。
该门禁证明的是“把已持久化 baseline 应用于一场赛前比赛”的可信边界；`TeamBaselineArtifact`
自身仍由上游构建器产出，尚未从版本化 canonical 球队过程事实和 raw parser contract 独立重算，
因此不能据此宣称整条 baseline 生产链已经完成来源字节级认证。

同一 feature spec 的比赛上下文只接受 `match-context-input/2`。专用 writer 从 canonical 精确赛程
版本、可重放的 schedule raw 和 typed 90 分钟结果按 cutoff 分别计算主客队上一场与休息天数。
用于寻找上一场的赛程证据必须同时满足：来源球队集合精确等于已注册球队、比赛数等于
`n * (n - 1)`、每个有向主客对阵恰好出现、无 parser diagnostics，且整份 raw 已在 cutoff 前观察到
或每行 fixture 都有不晚于 `as_of` 的 `known_at`。任一条件失败时双方 context 都保持 `missing`，
原因是 `season_schedule_coverage_incomplete`，不回退到更老比赛、不填零。snapshot 会把 context
精确绑定到 match/version/teams/`as_of`/kickoff，calibration `/2` 只修改对应一侧 lambda。通用
writer 不能新建 context `/1` 或 `/2`，旧 `/1` 仅保留 audit replay。该能力不等于 trusted
capture-run，也不证明经验系数已经由正式校准数据集和评审产物确认。

正式训练资格已使用 `training-qualification/1`。专用 writer 不接收调用方自报的 passed/reasons，
而是把 match/version、task qualification、ruleset、snapshot/result/fact refs 绑定后，从
`CanonicalFactStore.availability` 与 `assess_lifecycle` 重算结果；读取时按同一路径重放。training
dataset schema v2 只接受带该资格引用的 formal sample，旧 schema v1 与其自报 readiness 仅保留
audit 路径。team-baseline qualification 已能绑定并重放 typed result 和双方 typed team refs；通用
availability/lifecycle 会要求两条 latest team refs 共用 match/version/contract/raw/`observed_at`/
`known_at`，mismatch 以显式 reason fail closed，training qualification 按同一路径继承。captured 资格
在 trusted capture-run validator 缺失时仍 fail closed。player-profile qualification 已能从最新 verified
`MatchReportPlayerBatch` 重算实际阵容、角色、分钟和必需 metrics，绑定 batch contract 及全部 player/
actual-lineup fact refs；完整 22 人测试可以通过，缺失、无效或歧义 batch 则保留诊断并失败。该局部
资格通过不能据此宣称已有正式可训练的全季数据集。formal score sample、dataset、model run 和
prediction 固定使用当前
`snapshot-score-features/1` projection，不接受调用方自报其他 feature version。dataset/model-run
loader 还要求各自恰好一份 semantic manifest，并精确核对完整 payload、版本、input/output refs、
状态、质量和时间；CLI command wrapper 只登记命令输出文件引用，不再重复认领 dataset/model-run ID。

评估记录已升级为 `EvaluationRecord/2`，paired comparison 使用
`evaluation-comparison/2`。治理存储会从持久化 prediction/model/sample/canonical result 重算
Brier、LogLoss、score loss、固定 paired-bootstrap、reliability 和完整 cohort 的 `all` subgroup；
comparison 的 `cohort_ref` 会加载单独的 formal evaluation dataset，其 sample refs 必须精确等于该
dataset 的全部 included samples，且不得包含 train split。成功/partial model run 又必须有 eligible
train split，因此训练 dataset 不能被原样冒充正式 evaluation cohort；每对 challenger/champion
prediction 还必须在对应 typed result label 可知前生成。对两条模型训练 dataset 的全部 sample 身份、
match/version、snapshot、label 和 qualification refs 做显式不相交复核的强化门禁仍在本批收尾，
联合深回归前不得把该 cohort 解耦描述为生产验收完成。
可用市场基准仍必须引用真实 market snapshot 和 raw。score-model 的 `result-90/1` 标签会通过
typed `MatchResult90` loader 校验 canonical 内容 ID、精确 `fact_evidence`、RawArchive 注册/字节，
并要求 fact `observed_at` 等于 raw observation、标签 goals 与 `label_known_at` 精确一致；该 loader
不会从来源字节重新解析比分，也不会按版本化 normalization policy 重算 `known_at`。training
availability DAG 已要求成功/部分成功 dataset 的 `generated_at` 不早于样本的
`feature_known_at`、`label_known_at`、captured
`capture_observed_at`；每个 input reference 必须由 typed resolver 提供不晚于 `generated_at` 的
authoritative `available_at`，否则 fail closed。可提供 semantic `known_at` 的 feature ref 还必须
不晚于样本声明的 `feature_known_at`；captured sample 的每个 feature ref 还必须满足实际
`available_at <= sample.as_of`。通用 `derived-artifact` 会递归验证其 manifest 输入，拒绝循环、
无 authoritative availability 和晚于 artifact `generated_at` 的输入。model run 不得早于 dataset
生成，prediction 不得早于 model run 结束。上述检查尚未覆盖所有 canonical/derived 引用的语义
归一化，也不能替代 trusted capture-run，不能据此宣称完整无时间穿越或 prospective captured 已
完成。PaperBetLedger 默认仍为赛果构造 typed result validator；在另行提供 semantic market authority
的 accepted entry 中，带赛果的 settle/load/entries 会进入该 result gate，`result=None` 仅用于无需
赛果的 void 结算。

PaperBetLedger 的 accepted 路径现在要求 semantic market authority 按 snapshot ID 加载完整、已验证
的 `MarketSnapshot`，并与账本携带的对象逐字段精确比较。默认未注入 authority 时 accepted candidate
fail closed 为零仓位 rejected 记录；已有 accepted entry 的 load/entries/recompute/void 也不会绕过
原市场验证。`RawArchive` 只证明 raw 字节血缘，不能充当 semantic market authority；当前正向用例
使用的内存 registry 仅是测试替身，不是 operator-owned persisted market store。正式 market schema
v2 把 `period='90m'`、`market_type` 和 `line` 纳入内容身份：result market 禁止 line，totals/handicap
必须携带有限 line，账本结算规则必须与 authority 返回的 line 精确一致。accepted entry 还从已重放
snapshot 取得 `scheduled_kickoff_used`，要求 prediction `generated_at` 与 entry `placed_at` 都严格早于
开球；append/load/entries/recompute 均复核该门禁，不信任调用方另传 kickoff。

本地命令已覆盖不可变 command output、成功/partial/failed run manifest、显式 replay resume、
机器可读诊断和若干幂等恢复路径。覆盖 gate 从 canonical season/match mapping 与持久化 collection
attempt 计算；match-report 和 official-lineup contract 可在 derived 引用解析时重放。上述能力只
证明当前合同边界和两场离线 golden replay：实际覆盖为 2/20 支球队、2/380 场比赛，全部赛前
证据均为 `reconstructed`。golden schedule 不能证明完整双循环，因此 context fail closed；formal
dataset 的 train 与 holdout 两条 sample 均因赛前 snapshot/资格不满足而 excluded，eligible train 和
holdout 都为空。model gate 以 `no_eligible_train_split` 失败，prediction 与 evaluation 均
unavailable；这不是模型质量、真实市场基准或 ROI 证据，也不证明真实来源全季采集。

以下边界继续保持 Open：R02 下游 exact `mapping_id`/version pin（当前仅 official-lineup 与
match-report 的 ingest/replay 按 raw/contract `observed_at` 解析历史映射，其他下游不作此保证）、
aliases/candidate confidence policy、现有 resolver 自动入队整合、player 与 provider-scoped
Football-Data mapping、映射修订后的下游 requalification/recompute，以及完整跨源
生产验证；match-report player facts 之外的通用 `lineup_facts`、`news_evidence`、`prematch_events`
内容与证据重放 verifier；其他来源的 lineup/news/event adapter；通用 player-profile producer；
从来源字节推导比分和 `known_at` 的版本化
`result-observation/parser-normalization` contract；`file-sha256` 引用对应文件字节的定位和复验；
operator-owned、versioned `SourceRegistry`，以及行级 schedule `known_at` 元数据的生产签发与审计；
snapshot completeness vNext；通用 baseline builder 的 canonical-only 输入门禁；`observation_version`
对 SQLite `rowid` 顺序的剩余依赖；公开可构造 lifecycle DTO 的非 formal authority 边界；lineup-delta
校验完整贯穿 request-local session；`load_score_grid_composition_payload()` 的 orphan artifact manifest
authority；comparison 创建 CLI；
20 队/380 场及逐场真实 attempt；prospective `captured` capture-run；权威真实市场存储与 cohort；
paper ledger 的 prediction capture mode 持久化与 prospective ROI 分层；经评审的晋级政策和观察期。
市场、captured 或政策缺失时必须 fail closed，不得据此声称 ROI、模型晋级或全季完成。

## 执行原则与依赖

1. 先修证据、身份和 canonical 约束，再修快照、生命周期和训练资格。
2. 先完成可信的英超纵向切片，再用注册表和配置扩展赛事；不得复制联赛专属流水线。
3. 历史 `2025-26` 赛前信息只能标记为 `reconstructed`。`captured` 能力须从下一批尚未
   开赛的注册比赛开始前瞻验证，不能通过回填补造。
4. 市场缺失不删除足球预测；真实市场评估、模型晋级和纸面投注分别设置准入门禁。
5. 免费来源的失败、403、字段缺失和身份冲突都是正式结果，必须保留采集尝试和诊断。

依赖主线为：`R02/R04/R08 -> R07/R09 -> R01/R03/R10 -> R11/R05/R06 -> R12`。
其中 R06 的模型晋级证据必须等待 prospective captured 样本，不能靠实现完成来替代观察期。

## 修复项

### FDP-R01：快照与正式预测准入可由调用方伪造

- 严重度：P1；状态：Open（部分持久化准入合同已实现，snapshot completeness、可信上游 producer
  与 trusted capture-run 仍开放；不能视为 production accepted）。
- 当前实现证据：snapshot schema v2 已有严格 canonical parser/round-trip 校验；prediction schema
  v4 与 composition schema v2 的共享 loader 会重放 snapshot、model run、calibration、composition
  及 manifests，并被 training、governance 和 paper ledger 使用。正式 `prematch-features/3` 要求
  `team-baseline-input/3`；专用 writer 从 canonical 精确比赛版本和 persisted baseline artifact
  重算主客 lambda，且 baseline semantic manifest 必须唯一并精确匹配 writer contract，旧 `/1`、
  `/2` 只读。同一 spec 的 `match-context-input/2` 只有在 cutoff 前可证明完整双循环赛程时，才从
  canonical 精确版本和 typed result refs 分别重算双方上一场与休息天数；否则双方均显式 missing。
  snapshot 把 context 绑定到 match/version/teams/cutoff/kickoff。baseline 改值、换比赛/版本/方向/
  cutoff、calibration 参数改值及 ready lineup 贡献遗漏即使重新计算内容 ID 也会失败。成功/部分
  成功 training dataset 中，
  captured sample 的每个 feature ref 必须具有不晚于 sample `as_of` 的 authoritative availability；
  capture evidence 自身的 raw observation 也不得晚于 `as_of`。`training-qualification/1` writer 会
  从 canonical availability 与 lifecycle ruleset 重算 match/version/task-specific passed/reasons，
  formal dataset schema v2 再重放该资格；调用方自报 readiness 的 schema v1 只能进入 audit 路径。
  formal score projection 固定为 `snapshot-score-features/1`，dataset/model-run manifest 必须唯一且与
  对象精确一致；正式 prediction 的 `generated_at` 还必须严格早于 persisted snapshot 的 scheduled
  kickoff。
- 剩余缺口：snapshot 当前只允许 `reconstructed`，trusted capture-run validator 尚不存在；
  `prematch-features/3` 的 required feature set 仍沿用旧版：T-24h 未要求预计阵容与球员 availability，
  `lineups-confirmed` 也未要求可用替补或完整比赛名单，因此 `ready` 尚不等于设计要求的完整赛前
  信息；需要 snapshot completeness vNext 并将旧版本降为 audit-only。
  baseline application 已关闭调用方注入 lambda 的路径，但 baseline artifact 自身尚未从版本化
  canonical 球队过程事实与 raw parser contract 独立重算；因此不能宣称所有正式预测输入及其
  上游 producer 均已完成来源字节级认证。
- 设计依据：总体设计第 2、9、13、15 节；ADR-0013、ADR-0014、ADR-0018。
- 初始复现/根因：快照构造接受调用方提供的 `capture_mode`、`missing_fields` 和宽松 `source_ref`；
  readiness 与预测入口没有从持久化证据重新计算，空特征或历史运行可伪装为正式输入。
- 修复范围：由校验器根据 raw/canonical 引用、比赛版本、`known_at`、`observed_at`、运行
  时间和完整性配置计算快照模式与缺失项；只有赛前实际采集并在赛前不可变封存的运行
  能产生 `captured`。预测入口只接受已持久化、校验通过且版本一致的快照。
- 依赖顺序：先完成 R08 的证据/运行清单字段和 R04 的事实约束，再收紧快照及预测入口。
- 明确非目标：不把历史重建升级为实采；不以填零、默认引用或手工 `ready` 绕过缺失。
- 验收测试/证据：空特征、未知引用、赛后生成、`known_at > as_of`、伪造
  `capture_mode` 均失败；合法 prospective 采集可通过；产物可追溯到原始证据和成功运行。
- 完成定义：所有准入字段由版本化校验器计算，负向用例在单元与 CLI 端到端层均被拒绝，
  正式预测中不存在调用方可写的捷径。

### FDP-R02：Football-Data 与 FBref 跨源身份分裂

- 严重度：P1；状态：Open（schema v7 版本化映射与审计存储第一切片已实现；跨源生产闭环未完成）。
- 当前实现证据：schema v7 以原子迁移把 source mapping 升级为稳定 `mapping_id`、单调 version、
  `created_by`、`supersedes_mapping_id` 和 UTC 半开有效区间。SQL 约束、唯一索引和 trigger 会拒绝
  错误实体类型、版本跳跃、重叠区间、多个 current、越权更新/删除及不完整修订。冲突、revision
  event、decision、evidence 和 sibling obsoletion 均追加写入；未映射或低置信度候选可排入持久化
  队列。受控人工 override 使用 expected-current `mapping_id` 做 CAS，对完全相同的重试幂等，并
  保留理由、操作者和证据。resolver 支持 current、显式 version 和 UTC `as_of` 历史重放。fallback
  解析路径仍要求复用注册球队和已映射比赛，Football-Data 赔率列不会进入模型输入。
- 剩余缺口：下游 contract 尚未精确 pin `mapping_id`/version，当前只靠 `observed_at` 保持 legacy
  映射稳定；aliases 与 candidate confidence policy 尚未形成版本化契约，现有 resolver 也未自动
  整合冲突入队。player mapping、provider-scoped Football-Data ID 解析、映射修订后的 downstream
  requalification/recompute、既有重复数据迁移以及完整跨源生产验证仍未完成。
- 设计依据：总体设计第 7、17 节；ADR-0015、ADR-0017、ADR-0022。
- 初始复现/根因：Football-Data 适配器由展示名生成来源实体，下游直接创建球队和比赛，未先解析
  带审计的来源映射，因此同一球队和比赛被重复建档。
- 修复范围：建立 source ID -> platform ID 的版本化映射和冲突队列；别名只生成候选；
  fallback 赛程/赛果必须复用已解析的球队、比赛及比赛版本，改期不创建新 `match_id`。
- 依赖顺序：先稳定映射契约和数据库唯一/外键约束，再迁移适配器和既有重复数据。
- 明确非目标：不依靠模糊名称在下游自动合并；不让 Football-Data 赔率或统计进入模型。
- 验收测试/证据：当前负向测试已覆盖 schema v7 原子迁移、版本/区间/append-only SQL 约束、
  unmapped/低置信度冲突入队、CAS/幂等人工修订、sibling obsolete 和按 version/`as_of` 重放。完整
  验收仍须先导入 FBref 的 2 队/2 场，再导入 provider-scoped Football-Data fallback，证明 canonical
  数量不增加、赛果挂到原比赛，且修订后下游按精确映射版本重新资格判定和重算。
- 完成定义：所有 downstream join 使用 platform ID；重复导入幂等；修订映射可重算并保留
  旧版本及审计证据。

### FDP-R03：全赛季覆盖门禁存在假阳性

- 严重度：P1；状态：Open（覆盖与 context 的 fail-closed 合同已实现，20/380 生产证据未取得；
  不能视为全季 production accepted）。
- 当前实现证据：canonical schema v6 持久化并迁移 collection-attempt `source_id`、版本化
  prematch event 和 content-addressed match-report contract；覆盖校验从赛季映射和持久化
  attempt 计算逐 fixture 状态，错误 source ID、报告 URL 和 swapped fixture/report 身份不会
  满足 attempt 或 report-success gate，并保留可读 mismatch 诊断。另一个用于正式
  `match-context-input/2` 的赛程门禁会核对 registered team source IDs、`n * (n - 1)` 场、有向主客
  对阵全集、parser diagnostics 和整份/逐行赛程在 cutoff 前可知性；任一缺口使双方 context 都以
  `season_schedule_coverage_incomplete` 保持 missing，不会跳过遗漏的中间 fixture 后回退到更老比赛。
- 剩余缺口：当前 golden 只有 2/20 支球队和 2/380 场，两条 formal sample 均 excluded，不能证明
  全季赛程或逐 fixture collection attempt。真实 20 队双循环、380 场唯一比赛、每场持久化 attempt
  及最终状态仍未取得；行级 fixture `known_at` 也缺 operator-owned 来源签发与生产审计。
- 设计依据：总体设计第 17 节；ADR-0004、ADR-0013、ADR-0017；`AGENTS.md` 全赛季边界。
- 初始复现/根因：门禁只比较数量和调用方提供的 attempt ID，没有验证双循环赛程结构，也没有查询
  持久化采集尝试。
- 修复范围：验证 20 支唯一球队、每队 38 场、每对球队主客各一次、380 个唯一比赛身份；
  每场从 evidence store 查询至少一次采集尝试及最终状态，缺失/阻断保持可见。
- 依赖顺序：依赖 R02 的稳定身份、R04 的赛程事实约束和 R08 的采集尝试清单。
- 明确非目标：不要求受阻来源伪装成功；不以两场 golden 或 380 个任意 ID 代表全赛季。
- 验收测试/证据：只有 10 个唯一对阵、重复主客场、虚构 attempt ID、球队场次不平衡均
  失败；真实报告列出 20/380、每场 attempt 状态、缺失项和最后尝试时间。
- 完成定义：赛季门禁只能从持久化事实与采集证据计算，任何结构或尝试缺口均返回非零
  退出码和机器可读诊断。

### FDP-R04：canonical 事实约束与生命周期推进过宽

- 严重度：P1；状态：Open（typed result/team/player-batch replay 与局部 lifecycle/training 门禁已实现，
  result normalization、其余来源/事实 verifier、历史重认证和若干信任边界开放）。
- 当前实现证据：结果、球队观察、球员观察和阵容写入会检查 finished match version、比赛参与方
  与球员 assignment；三种 readiness 独立计算。`load_verified_match_result` 会重算 typed
  `MatchResult90` 的内容 ID、精确 `fact_evidence`，并核对 RawArchive 注册元数据和实际字节；fact
  `observed_at` 还必须等于 raw observation。`load_verified_team_observation` 进一步从唯一 match-report
  contract 和 raw parser 重算球队 `aggregated_stats` 与对应主客进球，核对合同/比赛版本/球队映射、
  collection attempt、raw ID、`known_at`、`observed_at` 和内容身份。availability 按 as-of 同时过滤
  knowledge/observation 时间，只选择每队最新候选；最新候选无效时保留 typed replay diagnostic，
  不回退旧版本。双方 latest observation 的 match/version/contract/raw/`observed_at`/`known_at`
  provenance 会被 availability 比较并由 lifecycle 独立重算；mismatch 保留两条 latest refs，记录
  `typed_team_fact_pair_mismatch`，且不回退旧组合。`archived-complete` 要求 typed result ref、配对一致
  的双方 typed team refs、完整 verified player batch 且没有 replay diagnostics；`team-baseline-ready`
  同样从 result/team refs 计算，score-model readiness 保持独立。match-report player batch 会从 raw
  parser 和历史映射重算全部 `PlayerMatchObservation` 及 `ActualLineupFact(official=False)`，以双 cutoff
  选择 latest，拒绝无效 latest 和同观察时点歧义，并为 formal `player-profile-ready` 绑定精确 contract/
  fact refs；完整 22 人样本可通过 lifecycle 与 training qualification。official lineup contract v2
  另行重放赛前双方官方首发，不能与赛后 actual lineup 混用。
- 剩余缺口：typed result 尚未绑定版本化 `result-observation/parser-normalization` contract；
  `load_verified_match_result` 不会从 raw 重新解析比分，也不会按 parser/normalization policy 重算
  `known_at`。match-report/official-lineup 合同之外的通用 `lineup_facts`、`news_evidence` 和
  `prematch_events` 仍无同等级 loader，其他来源 lineup/news/event 也未接入。team/player observation
  的 `observation_version` 不在内容 ID 中；UPDATE trigger 和连续版本检查关闭了直接调大 ordinal 的
  路径，但历史顺序检查仍依赖 SQLite `rowid`。公开可构造的 `MatchAvailability` /
  `PlayerObservationAvailability` 可直接传给纯 `assess_lifecycle()`；该入口只验证结构自洽，不能作为
  canonical authority。formal `TrainingArtifactStore._recompute_training_qualification()` 会从 canonical
  重算并精确比较，因此这是 P2 非 formal 信任边界，不是 formal training 绕过。team availability 的
  单条失败 candidate 诊断仍较粗且不保留 ref；历史数据完整重新认证和差异清单也未完成。
- 设计依据：总体设计第 6、9、13 节；ADR-0013、ADR-0014、ADR-0016。
- 初始复现/根因：结果、球队统计、球员观察和阵容缺少比赛状态及参与方约束；空统计集合也可满足
  `archived-complete`，导致错误事实继续进入 readiness。
- 修复范围：结果只能关联正确的已完赛比赛版本；球队必须是主/客参与方，球员须通过
  当时注册/阵容属于对应球队；完整性配置明确必填字段和非空基数；三种资格独立计算。
- 依赖顺序：先落数据库/领域约束和迁移，再修生命周期计算，最后重新认证已有数据版本。
- 明确非目标：不要求一场比赛同时通过全部三种资格；缺失与不适用不转换为零。
- 验收测试/证据：未完赛结果、第三队统计、错误球员、空统计归档全部被拒绝；合法部分
  数据只通过相应 readiness，并记录其他资格的排除原因。
- 完成定义：非法事实无法写入或推进状态；生命周期完全由版本化完整性配置计算；历史
  重新认证结果和差异清单可审计。

### FDP-R05：球队基线放大进球期望，三层贡献未进入唯一比分模型

- 严重度：P1；状态：Open（组合、baseline application 与防篡改契约已实现，baseline producer
  重算和正式校准证据开放）。
- 当前实现证据：team-baseline v2 使用明确的中性坐标并只应用一次场地项；
  `expected-goals-composition/2` 把 baseline、context 和所有 ready lineup dimensions 以唯一贡献键
  组合进 prediction v4。不可变 calibration policy 约束公式、source path/value、reference、
  coefficient 和 bounds，所有市场视图从同一 `DixonColesGrid` 重算。`team-baseline-input/3`
  从 persisted baseline artifact 与 canonical 精确比赛版本重算主客 lambda，并把 artifact、transform、
  比赛方向、开球和 cutoff 一起绑定；baseline loader 要求恰好一个与 payload、schema、transform、
  input/output refs、状态、质量和时间精确一致的 semantic manifest，任意 lambda 或 manifest 即使
  重哈希也不能进入正式 `/3` snapshot。`match-context-input/2` 仅在完整双循环赛程可由 cutoff 前
  证据证明时，从 canonical 精确 schedule versions 与 typed result refs 分别重放双方上一场和 rest
  days；覆盖不完整时双方都显式 `missing`，context calibration `/2` 只作用对应一侧 lambda。golden
  vertical producer 不再直接消费 parser DTO，而是加载两条 verified `TeamMatchObservation`，要求它们
  属于同一 match/version/contract/raw/`known_at`/`observed_at`，并把两条 fact refs 写入 baseline lineage。
- 剩余缺口：`TeamBaselineArtifact` 的球队强度和输入观察仍由上游构建器提供，当前 `/3` 门禁不会
  从 raw 比赛报告和版本化 canonical 球队过程事实重新拟合整个 artifact；因此它能证明 baseline
  的应用公式与比赛绑定，不能证明 artifact producer 已完成逐字段 raw replay。通用
  `build_team_baseline`/`TeamMatchProcess` 仍可由调用方提供数值与看似真实的 refs；本次 canonical-only
  约束只覆盖 golden vertical producer，不能外推为 generic baseline builder 已闭环。当前 policy 系数也
  缺少可追溯经验校准数据集与评审产物，只能证明公式可重放，不能证明参数已经完成正式校准。
- 设计依据：总体设计第 11、12 节；ADR-0003、ADR-0007、ADR-0008。
- 初始复现/根因：球队均值所处坐标、场地倍率的估计口径和二者组合方式不可验证；已复现的
  对称样本由主客均值 1.5/1.0 得到 1.875/1.25，显示输出被放大且校准口径不可验证。
  纵向流水线又只把阵容和上下文写入快照，最终 lambda 未消费这些贡献。
- 修复范围：先明确并版本化基线是中性坐标还是主客条件坐标，再用训练/校准证据确定场地项
  的组合方式；以版本化贡献键组合球队基线、阵容增量和比赛上下文；输出 lambda 分解，并
  让全部盘口聚合同一 `DixonColesGrid`。
- 依赖顺序：依赖可信 readiness、球队/球员事实和版本化特征产物；在 R06 正式评估前完成。
- 明确非目标：本项不引入未经验证的机器学习修正层，也不以市场赔率校准足球模型。
- 验收测试/证据：固定合成样本和历史校准集能解释从基线坐标到 lambda 的每一步，且不会
  出现未在版本化公式中声明的放大；重复贡献键失败；修改阵容或上下文会按可解释幅度改变
  lambda；所有市场视图与同一网格逐项守恒。
- 完成定义：坐标定义、校准方法和公式均有版本及回归证据；三层真实进入最终 lambda；预测
  保存贡献明细、特征版本及唯一网格引用。

### FDP-R06：真实市场评估与模型晋级门禁可绕过

- 严重度：P1；状态：Open（部分 evaluation/governance/market 合同已实现，真实市场与晋级观察
  闭环未生产验收）。
- 当前实现证据：`EvaluationRecord/2` 和 `evaluation-comparison/2` 会从持久化 prediction、model
  run、training sample 与 typed canonical result 重算指标、固定 paired-bootstrap、reliability
  和完整 cohort 的 `all` subgroup；未来 model run、伪造概率/赛果/汇总和 cherry-picked subgroup
  均会失败。comparison 会加载单独 formal evaluation dataset 的全部 included samples，并禁止 train
  split；成功/partial model run 的训练 dataset 必须包含 eligible train，因此不能把训练 dataset
  原样当评估 cohort。当前还在收紧对 challenger/champion 训练样本身份与引用的显式不相交门禁，
  本批联合深回归未完成前只算合同实现进行中。缺市场不会删除足球预测，benchmark 保持 unavailable。
  正式 market snapshot
  schema v2 把 90 分钟 period、market type 和 line 纳入身份；result 禁止 line，totals/handicap 要求
  有限 line。PaperBetLedger 的 accepted 路径要求 authority 按 ID 返回完整 verified market snapshot
  并精确比较；默认无 authority 时 fail closed，`RawArchive` 不能替代 semantic lookup。
- 剩余缺口：没有 operator-owned authoritative persisted market snapshot store；当前内存 registry
  仅用于测试，derived resolver 对生产 market refs 继续 fail closed。也没有真实 timestamped market
  cohort、从已持久化 evaluation pairs 创建 comparison 的 CLI、trusted capture-run validator，或
  仓库内可证明已评审并完成观察期的 promotion policy；当前 reconstructed 合同证据不能建立
  prospective 模型质量或 ROI。
- 设计依据：总体设计第 14、15、17 节；ADR-0008、ADR-0009、ADR-0018、ADR-0019。
- 初始复现/根因：评估接受未来报价、未知 raw 引用和不完整市场；治理对 NaN/空样本比较失效，且
  缺少经评审、版本化的晋级政策与 prospective captured 样本约束。
- 修复范围：市场快照验证真实来源、原始证据、互斥选项完整性、有限赔率、状态和时间；
  captured/reconstructed 分层评估；所有非有限指标拒绝晋级；晋级政策保存评审、最小样本、
  观察期、置信方法、分组退化限制和回滚目标。
- 依赖顺序：依赖 R01 的真实 captured 证据、R05 的一致模型输出和 R08 的版本化评估产物。
- 明确非目标：不保证盈利，不用合成赔率或短期 ROI 证明模型质量，不连接真钱账户。
- 验收测试/证据：未来报价、未知证据、不完整市场、NaN、空 cohort 均使 benchmark 或
  promotion 失败；缺市场时足球预测仍生成并标记 benchmark unavailable；合法 captured
  cohort 的报告可复算 Brier、LogLoss、校准和去水市场比较。
- 完成定义：无版本化评审政策或前瞻样本时挑战者只能影子运行；每次晋级/拒绝均保存一致
  样本、指标、分组诊断、置信证据和可回滚产物。

### FDP-R07：FBref 逐场归档链路尚未证明适配真实报告

- 严重度：P1；状态：Open（parser/fixture/result 及 match-report team/player stats replay 已实现，真实
  FBref 变体适配、完整生产采集和全季证据开放）。
- 当前实现证据：match-report 解析在写事实前交叉校验 raw 页面、赛事/赛季、双方、日期、比分、
  URL 和来源身份；生产导入固定要求七张表，显式自定义表集合仅用于 preview/研究，不能满足
  完整报告覆盖门禁；强门禁会从 RawArchive 重放并核对 parser version、source match、主客方向、
  赛事/赛季、日期、90 分钟比分、逐队表集合、缺表和阻断诊断；缺失表、重复表、未知表及逐队
  缺表均形成结构化诊断并阻断相应事实写入。赛程 parser 还会保留每行
  `data-fdp-fixture-known-at`，显式解析 cancelled/postponed，并允许 context 按精确 canonical match
  version 重放对应 schedule raw；table-level 时间不能认证单场版本。typed team observation 与 player
  batch loader 已把写出的 canonical team/player stats、实际 starter/bench 集合和精确 evidence 逐字段
  绑定回同一 raw parser 输出，删除、篡改、错配历史映射或不完整 batch 会失败。该证据来自保存的
  离线 fixture；尚未证明 parser 能覆盖真实 FBref 报告的生产变体、受阻响应和全季 380 场，真实
  成功/失败 attempt 清单也未取得。
- 设计依据：总体设计第 5.2、16、17 节；ADR-0004、ADR-0017、ADR-0022。
- 初始复现/根因：比赛报告管线信任调用方比赛 ID/比分，未与 raw 页面身份交叉核验；解析器依赖
  golden fixture 人工提供而真实 summary 可能缺少的字段。
- 修复范围：从 raw 证据解析并核对赛事、双方、日期、比分和来源 match ID；按真实 schema
  显式处理可选字段与表格变体；逐表保存解析诊断、版本和缺失原因；每场持久化采集尝试。
- 依赖顺序：依赖 R02 身份解析和 R08 raw/attempt 清单，再推进 R03 全赛季门禁和 R04 归档。
- 明确非目标：不绕过 Cloudflare/访问控制，不用 Football-Data 替代 FBref 球队/球员统计。
- 验收测试/证据：由真实保存页面裁剪的离线样本覆盖 summary、passing、defense、possession、
  misc、keeper 等约定表；身份/比分不一致失败；在线 403 返回持久诊断和退出码 3。
- 完成定义：英超 380 场每场都有 attempt；成功报告除 parser/fixture/result contract 外，还能从
  raw 逐字段重放并核对写出的 canonical team/player stats；失败报告保持 pending 且可补采，不能
  因 golden 解析成功宣称全季完成。

### FDP-R08：derived 血缘、运行清单和静态报告未闭环

- 严重度：P2；状态：Open（manifest、typed resolver 与直接字节证明已部分实现，深层闭环开放）。
- 当前实现证据：derived artifact/run manifest 校验 namespace、内容身份、时间、状态、input/output
  refs 和重复 output 的语义一致性；失败/partial 运行、command output、training/model、governance、
  paper ledger、match-report 与 official-lineup contract 已有相应持久化或 replay 路径。training 的
  通用 `derived-artifact` resolver 会递归遍历 manifest `input_refs`，只接受 succeeded/partial，
  并拒绝循环、无 authoritative `available_at` 或晚于 manifest `generated_at` 的输入；write/load
  使用同一门禁。`match-context-input/2` 会把目标及上一场 schedule raw、及时 typed result refs 和
  cutoff 写入 content-addressed lineage，读取时从不可变字节和 canonical 精确版本重新计算。
  formal dataset、model run 和 team baseline loader 都要求对应 semantic output 恰好由一份 manifest
  认领，并逐项核对 writer contract 的 payload、schema/transform、input/output refs、状态、质量与
  时间。`register-training-dataset` 和 `register-model-run` 的 CLI wrapper manifest 只认领命令摘要/
  文件内容引用，不再重复把 dataset/model-run ID 列为自身 output；semantic manifest 的唯一所有权
  因此不会被通用命令 wrapper 破坏。request-local `VerificationSession` 通过 `ContextVar` 在同一验证
  请求内传播，提供单一只读 canonical SQLite snapshot、tri-state entity identity lookup、fail-closed
  manifest/sample catalog 和带 close-time 终检的 `FileProof`。当前接入路径只在本次请求内缓存已完成
  的 artifact manifest、typed result、typed team/player observation、actual-lineup、player batch 与
  match-report replay；同一 report contract 不重复解析，失败不作为可复用验证结果。raw manifest 和
  对象字节均进入 `FileProof`，canonical result/team/player/report replay 使用同一 DB snapshot；manifest
  的 in-progress guard 会拒绝递归 cycle。
- 剩余缺口：上述 checkpoint 只证明已接入路径在单请求内使用一致 DB 视图并复验已读文件，不等于
  全链闭环。model/dataset、match-context、official-lineup、audit 旧扫描链，以及其他 public raw/
  derived loader 尚未全部贯穿同一 session；cache 也不是跨请求 authority。尤其
  `DerivedArchive._validate_lineup_delta_source()` 对 raw source 和 player profile 的嵌套加载未传递
  `verification_session`。public `load_score_grid_composition_payload()` 虽有文件证明和 content ID 校验，
  仍不要求唯一、成功且类型正确的 semantic manifest，合法 orphan bytes 仍可能被直接加载。
  通用 `file-sha256` resolver
  仍只校验引用语法，不能定位并
  重新哈希任意对应文件，也不能描述为可离线取回。typed result resolver 虽核对内容 ID、精确
  evidence、RawArchive 注册/字节和 fact/raw `observed_at`，仍缺版本化 parser/normalization contract
  从 raw 重算比分与 `known_at`；通用 lineup、news 和 prematch event 仍缺 typed replay verifier。
  行级 schedule `known_at` 虽可重放，其生产签发和 operator-owned 来源治理
  尚未建立，不能把 bundled fixture 元数据当作 prospective capture 证据，也不能据此证明全季、
  captured、ROI 或模型质量。
- 设计依据：总体设计第 6.3、16、17 节；ADR-0012、ADR-0016。
- 初始复现/根因：归档重点限于 snapshot/prediction，球队基线、球员画像、训练集、模型运行、
  评估等缺少统一产物登记；运行摘要缺少完整时间、参数、代码版本、输入输出和失败记录。
- 修复范围：为所有 derived 类型建立不可变产物清单；run manifest 保存参数、开始/结束 UTC、
  代码/解析器版本、输入输出引用、状态、错误和恢复点；静态报告只从版本化结构化数据重建。
- 依赖顺序：先提供 R01/R03/R07 所需最小 evidence/run 契约，再覆盖全部派生产物和报告。
- 明确非目标：不建设 Web 服务、公开 API 或以 HTML 作为事实来源。
- 验收测试/证据：相同固定输入重跑得到相同语义输出和内容引用；失败运行也有 manifest、
  非零退出码和可恢复位置；任一报告数字可反向追到数据、特征、模型和市场版本。
- 完成定义：所有正式命令均幂等、可恢复并产出 manifest；训练集、基线、画像、模型、评估、
  纸面账本和报告全部进入 derived registry，离线可重建。

### FDP-R09：赛前事件证据、确认时间与真实来源采集门禁缺失

- 严重度：P1；状态：Open（事件与 official-lineup 部分契约已实现，来源治理开放）。
- 当前实现证据：新闻 canonical 记录强制匹配 raw source/URL/`observed_at`；事件证据必须来自
  注册来源，确认级别由 official/独立来源规则计算，且满足
  `max(published_at) <= known_at <= max(observed_at)` 和 `known_at <= as_of`。赛前 attempt
  复用幂等 canonical ledger 并可保留失败 raw body/诊断；球员级事件持久化可见 match version，
  并按事件 `known_at`/`as_of` 截断同版本 assignment；v3 legacy 未绑定球员事件迁移后不可修改
  特征且保留原证据。官方阵容 JSON schema 2 会 raw-first 归档，并通过 contract v2 重放来源
  比赛、球队、22 个球员绑定及双方首发事实。
- 剩余缺口：`SourceRegistry` 仍是调用方可构造、可注册的内存对象，不是 operator-owned、
  versioned、content-addressed 持久化配置；调用方仍可临时注册 `official=True` 的来源并让
  official-lineup contract v2 接受该标签。contract v2 能证明 raw/parser、比赛/球员映射和 canonical
  首发事实一致，但不能证明来源的 operator authority。真实新闻、伤停、停赛和双方官方阵容持续
  collector/健康运行未证明，prospective capture-run 也尚未实现。
- 设计依据：总体设计第 5.3、8、9、17 节；ADR-0004、ADR-0005、ADR-0018。
- 初始复现/根因：调用方可用单条新闻把事件标记为 `corroborated`，事件 `known_at` 未与支持证据的
  `published_at/observed_at` 交叉校验；当前纵向切片主要依赖人工 fixture，没有可持续的
  新闻、伤停、停赛和官方阵容来源适配器。
- 修复范围：确认等级由证据规则计算：`official` 需要可验证官方来源，`corroborated` 需要
  至少两个独立来源；事件 `known_at` 不得早于达到对应等级的最后一条证据发布时间，且不得
  晚于观察时间。建立来源注册、raw 归档、实体解析、采集 attempt 和健康诊断，并实现真实
  新闻/伤停/停赛与双方官方阵容适配器。
- 依赖顺序：依赖 R02 的身份映射、R04 的事实约束、R08 的 evidence/run 契约和 R07 的通用
  采集诊断；完成后由 R01 快照和 R10 阵容/画像消费。
- 明确非目标：不把自动摘要当事实，不让单一传闻修改特征，不绕过登录/验证码，也不把
  “未采到”解释为球员健康或阵容不变。
- 验收测试/证据：单一新闻声明 corroborated、同源重复报道、`known_at < published_at`、未知
  raw/实体引用均失败；官方单源和两个独立来源按其真实时间通过。未来比赛保存两类阵容
  attempt、原始页面、双方 11 人身份、发布时间与 captured/reconstructed 判定。
- 完成定义：确认状态、`can_modify_features` 和首次可知时间全部由版本化规则计算；四类真实
  来源适配器可幂等运行并保留失败；任何进入快照的事件可回溯到满足门禁的原始证据。

### FDP-R10：球员画像与阵容增量可把空维度标记为 ready

- 严重度：P1；状态：Open（typed match-report observation、readiness、vertical profile 与 lineup-delta
  局部契约已实现，通用画像 producer、session 闭环和真实全季覆盖开放）。
- 当前实现证据：画像 readiness 校验角色必需维度、N/A、分钟/比赛阈值、窗口和 `as_of`；
  lineup delta 校验双方 11 人、非空 profile refs、契约版本、样本量、输入 raw/canonical
  引用和非空适用维度；`lineup-delta-input/3` 会逐项加载 player-profile manifest、重算增量，
  并与 `official-lineup-input/2` 绑定的 canonical official lineup facts 及其 raw refs 比对。
  球员/球队归属由 canonical lineup fact 边界校验。match-report `PlayerMatchObservation` 与
  `ActualLineupFact(official=False)` 已从 raw parser/history mapping 组成可重放 batch；完整 22 人测试
  能通过 `player-profile-ready`。bundled vertical profile 只消费 verified canonical player refs，且
  `played_at` 来自 canonical 精确版本 kickoff。
  `prematch-features/1` lineup 仅作为只读 legacy artifact；`prematch-features/2` 的 ready
  lineup 只接受 `/3` 证据。缺失、N/A、insufficient-sample 与 ready 保持独立。
- 剩余缺口：typed batch 只覆盖 match-report 来源；canonical `metrics_json` 仍是宽泛指标字典，角色化
  画像窗口和其他来源事实尚未形成完整生产合同。vertical producer 的 canonical-only 路径不能替代
  通用 `build_player_profiles` producer 的独立 canonical 重算门禁；真实全季球员覆盖也未证明。
  `DerivedArchive._validate_lineup_delta_source()` 的嵌套 raw/profile 加载未贯穿调用方
  `VerificationSession`，request-local 一致读边界仍需补齐。
- 设计依据：总体设计第 10、11、13、17 节；ADR-0006、ADR-0007、ADR-0014。
- 初始复现/根因：画像 readiness 主要按总分钟判断，即使 `metrics` 为空也可 ready；阵容的所有球员
  `dimensions={}` 时会得到空 delta 且 ready。长期能力、近期状态、可用性和负荷也尚未作为
  独立、带窗口和角色契约的产物表达。
- 修复范围：为角色版本定义必需、可选和 N/A 维度以及最小分钟/比赛数；画像分别保存长期
  能力、近期状态、可用性和负荷的窗口、样本与 `as_of`；阵容增量验证双方 11 人、参考阵容、
  角色适用维度和输入画像，空适用维度只能是 preview/not-ready。
- 依赖顺序：依赖 R04 的实际阵容/球员观察约束、R07 的逐场球员统计和 R09 的伤停/官方阵容；
  完成后供三种 readiness、R05 和 R11 使用。
- 明确非目标：不构造跨位置单一综合分，不把 N/A 或缺失填零，不因球员数据缺失删除球队
  基线，也不要求所有角色使用相同指标。
- 验收测试/证据：空 metrics、空 dimensions、缺必需角色维度、未来事实、小于样本门槛均不
  ready 并有原因；N/A 仅在角色配置允许时通过；窗口边界、per-90 样本分钟和角色分位可复算；
  完整 11 人阵容产生非空、版本化且可追溯的维度 delta。
- 完成定义：`player-profile-ready` 与 lineup delta 分别由版本化契约计算；四类球员状态独立
  保存；不存在空维度 ready 产物，报告明确区分 missing、N/A、insufficient-sample 和 ready。

### FDP-R11：版本化训练数据集与模型运行仍缺完整 typed availability DAG 和生产证据

- 严重度：P1；状态：Open（formal qualification、current projection、dataset/model manifest 与
  部分 availability DAG 合同已实现，完整语义重算和生产运行未验收）。
- 当前实现证据：`training-qualification/1` 绑定 match/version、task qualification、ruleset、
  snapshot/result/fact refs；writer 从 `CanonicalFactStore.availability` 和 `assess_lifecycle` 重算
  passed/reasons，读取及 formal dataset 写入时再次重放，不接受调用方自报资格。training dataset
  schema v2 只接受正式资格绑定；schema v1、自报 readiness 及其 model run 只能 audit，正式模型和
  prediction consumer 会拒绝。content-addressed dataset/model-run artifacts 还保存 capture mode、
  split、feature/label 版本、排除原因、模型字节、输出 hash 与评估 cohort。score-model 的
  sample/dataset/model-run/prediction 正式 feature contract 固定为
  `snapshot-score-features/1`；调用方传入其他 projection 会被拒绝。dataset 与 model-run loader
  分别要求其 semantic ID 恰好只有一个 manifest owner，并精确核对 payload、schema/transform、
  code/input/output refs、状态、质量与时间；CLI wrapper 只登记文件内容引用，不能成为第二 owner。
  `result-90/1` label ref 必须解析为通过内容/evidence/RawArchive 字节和 observation 时间校验的 typed
  `MatchResult90`，并精确匹配 goals 与 `label_known_at`。team-baseline qualification 还要求 availability
  选中的双方 typed `TeamMatchObservation` refs，并把 result 与 team refs 固定进 qualification
  `fact_refs`；最新可见 team candidate 无效时记录诊断且不回退旧事实。player-profile qualification
  同样从 latest verified player batch 重算并绑定 contract/player/actual-lineup refs；完整 22 人矩阵可
  通过，latest invalid 或 ambiguous batch 会 fail closed。dataset `generated_at` 不得早于
  每个样本的 `feature_known_at`、`label_known_at` 和 captured `capture_observed_at`；成功/部分成功 dataset
  的每个输入还必须具有不晚于 `generated_at` 的 authoritative `available_at`。当前能提供 semantic
  `known_at` 的 feature ref 会与样本 `feature_known_at` 交叉校验；captured sample 的每个 feature
  ref 还必须满足实际 `available_at <= sample.as_of`。通用 `derived-artifact` 会递归复用 typed
  resolver，要求每层输入不晚于所属 manifest `generated_at`，并对循环或无权威时间 fail closed。
  model run 不得早于 dataset 生成，且 `model_run.ended_at <= prediction.generated_at`。prediction
  schema v3 仅可 audit，不能被重写或晋级为当前合同。golden 的 schema v2 dataset 因完整赛程
  context 与赛前 snapshot/资格门禁，将 train 和 holdout 两条 sample 均明确 excluded；eligible
  train/holdout 都为空，model run 因 `no_eligible_train_split` 失败，因而没有
  prediction/evaluation。
- 剩余缺口：并非所有 canonical/derived reference 都有 typed resolver 可提供 semantic
  `known_at`；通用 derived manifest 的递归 availability 校验不会从任意 payload 推导语义时间。
  尚未接入的通用 lineup/news/prematch-event facts、`file-sha256` 以及 result 的 parser/normalization
  语义仍未闭环；
  不具备 authoritative `available_at` 的引用会 fail closed，并不等于已经完成语义重算。
  player-profile-ready 的 match-report checkpoint 已可通过，但通用画像 producer、其他来源球员事实和
  全季正式样本仍未闭环；team-baseline qualification 已继承 latest team refs 的 provenance 配对门禁，
  但单条 team replay 失败 candidate 的诊断仍过于粗粒度且不保留该 ref。trusted capture-run 仍
  不存在，因此这些 training gates 只能称为 availability/serialization 骨架，不能把 reconstructed
  历史升级为 prospective captured，也不能单独证明训练集完全无泄漏；正式可训练的 full-season
  dataset，以及真实生产 champion/challenger 训练、复跑和回滚证据同样缺失。
- 设计依据：总体设计第 6.3、12.2、13、15、16、17 节；ADR-0008、ADR-0012、ADR-0014、
  ADR-0016、ADR-0018。
- 初始复现/根因：derived 主要归档快照和预测，没有独立训练数据集/特征集/标签清单和可回滚
  模型产物；固定字符串 `model_run_id` 不能证明训练样本、参数、代码或评估区间。
- 修复范围：训练集 manifest 保存逐样本 ID、capture mode、资格配置、特征/标签版本、时间
  切分、排除原因和内容哈希；模型运行保存训练集 ID、算法/参数、代码/环境版本、开始结束
  时间、随机性、模型文件、评估 cohort 与输出哈希；champion/challenger 引用这些不可变产物。
- 依赖顺序：依赖 R01/R04 的可信资格、R08 的 derived/run registry、R09/R10 的事件与画像
  契约；R05 消费训练/特征版本，R06 消费模型与评估 cohort。
- 明确非目标：不以成功序列化模型证明质量，不用随机时间穿越切分，不把 reconstructed 样本
  冒充 captured 晋级证据，也不强制使用 XGBoost。
- 验收测试/证据：相同输入可重建相同数据集哈希和样本清单；未来特征、资格不符和标签泄漏
  被排除并记录；篡改任一输入使校验失败；模型从 manifest 可离线复跑并定位全部评估输出。
- 完成定义：每次正式/挑战者运行都有不可变数据集、特征、标签、模型及评估引用；不存在
  无训练来源的正式模型 ID；可按版本重现预测并回滚 champion。

### FDP-R12：纸面投注账本仍缺权威市场与生产结算证据

- 严重度：P1；状态：Open（账本与结算的部分 fail-closed 合同已实现，真实市场生产闭环和
  prospective ROI 证据未验收）。
- 当前实现证据：paper ledger 以 append-only revision 保存候选/拒绝、预测、model run、market、
  风险配置、stake 和结算；支持 win/loss/push/half-win/half-loss/void 规则，从 typed canonical
  result 结算，并可重算余额与敞口。默认 `PaperBetLedger` 只为赛果绑定 CanonicalFactStore/RawArchive，
  不再把 RawArchive 当作市场 authority；未注入 semantic market store 时，accepted candidate fail
  closed 为零仓位 rejected 记录，而 unavailable/rejected 记录仍可持久化。已有 accepted entry 的
  append/load/entries/recompute 和 void revision 均要求原 market snapshot 由 authority 按 ID 加载、
  验证 raw lineage 并与账本对象精确一致。append/load 使用共享 persisted-prediction verifier，summary
  artifact 和 recompute run 也能修复“entry 已写但 registry 尚未写”的幂等重试。正式 market schema
  v2 要求 `period='90m'` 并把 market type/line 纳入 content ID；result market 不得带 line，totals/
  handicap 必须带 line，结算规则的 line 必须与 authority snapshot 精确一致。accepted candidate 还
  从 persisted verified prediction snapshot 取得 kickoff，要求 prediction `generated_at` 与 entry
  `placed_at` 都严格早于开球；append/load/entries/recompute 都会再次执行该门禁。
- 剩余缺口：没有 operator-owned authoritative persisted market snapshot store、真实 timestamped
  报价 cohort 或生产纸面账本；当前正向测试依赖内存 registry，不是生产市场存储。result 语义也
  尚未由版本化 parser/normalization contract 从 raw 重算。ledger payload 也未持久化并验证 prediction
  capture mode，当前只能按 research/reconstructed 使用，不能形成 prospective ROI cohort。因此当前
  只能证明账本领域合同和 fail-closed 边界，不能证明真实赔率 ROI、运营质量或模型晋级。
- 设计依据：总体设计第 14、15、17 节；ADR-0009、ADR-0012、ADR-0019。
- 初始复现/根因：schema/报告骨架没有不可变逐笔账本，把当时预测、真实报价、风险配置、建议
  仓位、赛果和真实结算规则闭环；因此 ROI、敞口和结算均不可复现。
- 修复范围：账本记录候选生成/拒绝原因、预测与 model run、真实 market snapshot、风险规则、
  stake、同场/单日敞口、下单模拟时间、90 分钟赛果、结算规则版本及 win/loss/push/half/void
  结果；纠错以追加版本完成，并由静态报告引用账本版本。
- 依赖顺序：依赖 R04 的可信赛果、R06 的真实市场与准入、R08 的 manifest/registry、R11 的
  模型产物；在 Phase 3 只以影子模式积累和结算。
- 明确非目标：不连接账户、支付或自动下单，不把 Kelly 当组合相关性模型，不用纸面 ROI
  替代概率评分或触发模型晋级。
- 验收测试/证据：固定真实报价和规则样本覆盖赢、输、走盘、半赢、半输、作废及改期；重复
  运行不重复记账；未来/合成报价不能入账；逐笔余额、敞口和汇总可从不可变记录重算。
- 完成定义：每笔纸面决策从预测到结算均可追溯和复现，修订不覆盖历史；报告明确标记研究
  用途，且在市场或风险准入失败时只记录拒绝原因、不生成仓位。

## 分阶段路线图

### Phase 0：修复可信边界

- 状态：Open。R04/R05 及其他基础契约仍有上述明确缺口，本批全量 `pytest` 与统一质量门禁结果也
  尚未归档；局部分组测试或两场 golden replay 不能关闭本阶段。
- 顺序 1：完成 R02、R04、R08 的身份、事实、证据、运行清单和 derived registry 基础契约。
- 顺序 2：完成 R07、R09 的真实来源解析、新闻/伤停/停赛/官方阵容及采集 attempt 门禁。
- 顺序 3：完成 R01、R03、R10 的时间快照、赛季覆盖、画像/N/A 和 readiness 负向门禁。
- 顺序 4：完成 R11、R05、R06、R12 的版本化训练/模型、lambda 组合、市场治理和纸面账本
  契约；需要真实 prospective 样本的生产验收继续在 Phase 1 至 Phase 3 累积。
- 为每个复现用例增加负向回归测试；对既有 canonical/derived 数据重新认证，不静默修补。
- 出口证据：已复现的调用方绕过和假阳性全部关闭，十二项契约的负向测试通过；全量测试、
  lint、格式和编译通过；golden 重放两次语义输出一致，但仍明确标注为两场 replay。需要
  全季、真实来源或观察期的工作项保持 Open，直到后续阶段给出直接证据。

### Phase 1：完成 Premier League 2025-26 真实纵向切片

- 状态：Open。当前 bundled golden 仅覆盖 2/20 支球队和 2/380 场离线重放，全部赛前证据为
  `reconstructed`；它不构成全季覆盖、prospective capture、模型质量或 ROI 证据。
- 注册 20 队和完整双循环 380 场，并为每场持久化 FBref 逐场采集尝试、raw 归档或阻断诊断。
- 完成跨源身份、球队/球员赛后事实、比赛生命周期、三种独立 readiness 与排除原因。
- 完成新闻原始证据、结构化伤停/停赛事件、官方阵容适配器。2025-26 历史证据只能生成
  `reconstructed`；从下一批未来比赛开始真实封存 `T-24h` 和 `lineups-confirmed` captured。
- 按 R09 证明事件确认时间与独立来源门禁；按 R10 完成长短期画像、可用性、负荷和 N/A。
- 完成球队基线、阵容增量、比赛上下文、球员角色画像和 R11 版本化训练数据集；所有派生产物
  引用资格配置、样本清单、特征/标签版本和排除摘要。
- 出口证据：赛季 gate 从持久化数据计算且无假阳性；每场状态可解释；静态数据质量、球员
  分析和模型样本报告均可从 run manifest 指定的结构化版本重建。

### Phase 2：扩展完整数据范围

- 用同一注册表和流水线回填 Premier League、Bundesliga、LaLiga、Serie A、Ligue 1 的
  `2021-22` 至今数据；每个赛事按自身球队数和赛制配置覆盖门禁，不复制五套脚本。
- 纳入 Champions League、Europa League、Conference League、相关国内杯赛和国家队比赛，
  至少覆盖赛程、实际阵容/出场、分钟和负荷；统计深度不足必须显式标记。
- 持续运行新闻、伤停、停赛和官方阵容 prospective captured 采集，积累可用于正式评估的
  前瞻样本；历史缺少首次公开时间的事实保持缺失。
- 出口证据：五大联赛各赛季身份和覆盖报告、上下文赛事负荷关联、逐场 attempt 清单、raw
  血缘和缺失/冲突队列齐备；`2021-22` 至今范围逐项可审计。

### Phase 3：关闭模型、市场与治理闭环

- 以 R11 版本化训练集和模型产物运行 Dixon-Coles champion，并让 challenger 在一致的时间
  外样本影子运行。
- 接入独立真实市场来源，保存时间戳报价、原始证据、盘口/选项状态和去水方法；市场缺失
  只使 benchmark unavailable，不影响足球预测。
- 经评审固化 champion/challenger 晋级政策；captured 前瞻样本不足时保持 shadow，不晋级。
- 按 R12 建立不可变纸面账本，绑定预测、真实报价、规则版本、建议仓位、真实赛果和完整结算；
  输出概率评分、校准、市场比较和纸面运营指标，禁止合成赔率和真钱执行。
- 出口证据：评估可按 capture mode、联赛、主客、概率桶和完整性分组复算；晋级决策可回滚；
  纸面账本可按真实赔率和结算规则逐笔复现。

### Phase 4：第一阶段完成审计

- 所有生产命令具备幂等、断点恢复、结构化诊断、明确退出码和完整 run manifest。
- 生成引用明确数据、特征、模型、市场版本的静态数据质量、球员、模型与纸面账本报告。
- 按总体设计第 17.2 节逐条审计五大联赛身份、前瞻 captured、赛后完整性、三种 readiness、
  无泄漏训练集、一致样本评估、纸面账本和可重建报告；任何证据缺失即保持未完成。

## 全局非目标

- 第一阶段不提供滚球预测、自动投注、真钱执行、公开 API、Web/移动端或付费体育数据服务。
- 不以赛事模拟、短期 ROI、命中率、两场 golden 或“未发现错误”替代验收证据。
- 不绕过来源访问控制，不把来源失败解释为无比赛/无事件，不把缺失值静默填零。
- 不新增全局可写 `ready`，不混淆 `captured` 与 `reconstructed`，不让市场赔率进入足球模型。

## 统一验证与完成证据

每个阶段至少执行并归档以下命令的 run manifest：

```powershell
python -m pytest -q
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
fdp run-golden --observed-at 2026-07-16T08:00:00Z
```

此外必须保存对应阶段的负向门禁结果、全量覆盖报告、输入/输出内容引用和静态报告重建
校验。只有路线图内所有显式范围与总体设计第 17.2 节完成标准均有直接证据时，第一阶段
才可标记完成。
