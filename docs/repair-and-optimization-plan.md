# 修复与优化计划

- 状态：执行计划，不是验收证书
- 基线日期：2026-07-16
- 权威依据：[总体设计](superpowers/specs/2026-07-16-football-data-platform-design.md)、
  [ADR-0001 至 ADR-0022](README.md#架构决策记录)、[领域术语表](domain-glossary.md)
- 当前边界：先证明 Premier League 2025-26 纵向切片；两场 golden sample 只证明离线重放

本文把已复现的实现缺口转成可执行工作项和验收证据。状态只能由当前数据、运行清单、
负向测试和端到端产物共同证明；测试通过但未覆盖对应风险时，工作项仍保持未完成。

## Review Checkpoint (2026-07-17)

本节取代 2026-07-16 checkpoint 中“R01-R12 合同级修复均已完成”和固定测试数量等过强、
已过时的当前状态描述。当前工作树关闭了一批已复现的持久化伪造路径，但不是十二项全部完成，
也不是 Premier League 2025-26 全季或总体设计第一阶段的验收证书。最终测试数量和质量门禁结果
须在本批迁移、标签与时间校验修改全部落定后重新运行并记录，不在计划正文写死。

canonical 数据库已升级到 schema v6。官方阵容来源使用 JSON schema 2（parser
`official-lineup-json/1`）和 canonical official-lineup contract v2：raw 在解析前归档，来源比赛、
球队和 22 个来源球员绑定到平台 ID，双方首发、精确 `fact_evidence` 和 content-addressed contract
在同一事务写入。重放会重新解析 RawArchive 字节并核对当前映射、双方 11 人及 22 条 canonical
starter facts；legacy contract v1 和 DTO 写入路径均不能重新认证正式阵容。FBref match-report
contract 同样会从 raw 重放 parser、比赛身份、90 分钟赛果和生产表集合，summary-only preview
不能满足完整报告门禁。

正式预测已使用 prediction schema v4 和 score-grid-composition schema v2。持久化预测的共享
校验入口会重新加载并校验 snapshot、snapshot manifest、model run、prediction manifest、完整
calibration policy 和唯一 Dixon-Coles grid；training reference、governance comparison 与 paper
ledger 均复用该入口。贡献产物保存公式版本、来源路径/值、参考值、系数和 bounds，且正式预测
必须恰好消费 policy 规定的 baseline、context 和所有 ready lineup 维度。`legacy-inline/1` 不能
再创建正式预测；prediction schema v3 只能通过独立 audit-only view 读取，并明确保留 provenance、
calibration 和历史训练标签无法重新认证的限制。

评估记录已升级为 `EvaluationRecord/2`，paired comparison 使用
`evaluation-comparison/2`。治理存储会从持久化 prediction/model/sample/canonical result 重算
Brier、LogLoss、score loss、固定 paired-bootstrap、reliability 和完整 cohort 的 `all` subgroup；
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
完成。PaperBetLedger 默认构造路径会把带赛果的 settle/load/entries 送入该 typed result gate；
`result=None` 仅用于无需赛果的 void 结算。

本地命令已覆盖不可变 command output、成功/partial/failed run manifest、显式 replay resume、
机器可读诊断和若干幂等恢复路径。覆盖 gate 从 canonical season/match mapping 与持久化 collection
attempt 计算；match-report 和 official-lineup contract 可在 derived 引用解析时重放。上述能力只
证明当前合同边界和两场离线 golden replay，不证明真实来源全季采集。

以下边界继续保持 Open：R02 映射修订/冲突/override 账本；除 typed result 外其余五类 canonical
fact 的内容与证据重放 verifier；从来源字节推导比分和 `known_at` 的版本化
`result-observation/parser-normalization` contract；`file-sha256` 引用对应文件字节的定位和复验；
operator-owned、versioned `SourceRegistry`；从 canonical 赛程重算 `match-context-input/1`；
结构化球员单场观察与画像独立重算；comparison 创建 CLI；20 队/380 场及逐场真实 attempt；
prospective `captured` capture-run；权威真实市场存储与 cohort；经评审的晋级政策和观察期。
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

- 严重度：P1；状态：Open（持久化预测与 training captured-feature 时间门禁已实现，trusted
  capture-run 与 context 重算仍开放）。
- 当前实现证据：snapshot schema v2 已有严格 canonical parser/round-trip 校验；prediction schema
  v4 与 composition schema v2 的共享 loader 会重放 snapshot、model run、calibration、composition
  及 manifests，并被 training、governance 和 paper ledger 使用。baseline 改值、calibration 参数
  改值及 ready lineup 贡献遗漏即使重新计算内容 ID 也会失败。成功/部分成功 training dataset 中，
  captured sample 的每个 feature ref 必须具有不晚于 sample `as_of` 的 authoritative availability；
  capture evidence 自身的 raw observation 也不得晚于 `as_of`。
- 剩余缺口：snapshot 当前只允许 `reconstructed`，trusted capture-run validator 尚不存在；
  `match-context-input/1` 仍接受调用方计算的休息天数，尚未从 canonical 赛程重算，因此不能宣称
  所有正式预测输入均已关闭调用方伪造路径。
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

- 严重度：P1；状态：Open。
- 当前实现证据：当前 source mapping 表保存来源、来源 ID、平台 ID、有效时间、匹配规则、置信度
  和审计备注；fallback 解析路径要求复用注册球队和已映射比赛，Football-Data 赔率列不会进入
  模型输入。
- 剩余缺口：存储层没有关闭当前映射并创建 revision 的受控 API，也没有持久化冲突队列、人工
  override 的理由/证据账本和按旧版本重算能力；现有冲突主要直接失败，不能满足完整修订审计。
- 设计依据：总体设计第 7、17 节；ADR-0015、ADR-0017、ADR-0022。
- 初始复现/根因：Football-Data 适配器由展示名生成来源实体，下游直接创建球队和比赛，未先解析
  带审计的来源映射，因此同一球队和比赛被重复建档。
- 修复范围：建立 source ID -> platform ID 的版本化映射和冲突队列；别名只生成候选；
  fallback 赛程/赛果必须复用已解析的球队、比赛及比赛版本，改期不创建新 `match_id`。
- 依赖顺序：先稳定映射契约和数据库唯一/外键约束，再迁移适配器和既有重复数据。
- 明确非目标：不依靠模糊名称在下游自动合并；不让 Football-Data 赔率或统计进入模型。
- 验收测试/证据：先导入 FBref 的 2 队/2 场，再导入对应 fallback，canonical 数量不增加且
  赛果挂到原比赛；低置信度、同名和冲突映射进入人工复核并保留诊断。
- 完成定义：所有 downstream join 使用 platform ID；重复导入幂等；修订映射可重算并保留
  旧版本及审计证据。

### FDP-R03：全赛季覆盖门禁存在假阳性

- 严重度：P1；状态：契约已实现，生产证据开放。
- 当前实现证据：canonical schema v6 持久化并迁移 collection-attempt `source_id`、版本化
  prematch event 和 content-addressed match-report contract；覆盖校验从赛季映射和持久化
  attempt 计算逐 fixture 状态，错误 source ID、报告 URL 和 swapped fixture/report 身份不会
  满足 attempt 或 report-success gate，并保留可读 mismatch 诊断。
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

- 严重度：P1；状态：Open（写入约束和部分 replay 已实现，其余事实 verifier 开放）。
- 当前实现证据：结果、球队观察、球员观察和阵容写入会检查 finished match version、比赛参与方
  与球员 assignment；三种 readiness 独立计算。`load_verified_match_result` 会重算 typed
  `MatchResult90` 的内容 ID、精确 `fact_evidence`，并核对 RawArchive 注册元数据和实际字节；
  fact `observed_at` 还必须等于 raw observation。official lineup contract v2 会独立重放双方首发事实。
- 剩余缺口：typed result 尚未绑定版本化 `result-observation/parser-normalization` contract；
  `load_verified_match_result` 不会从 raw 重新解析比分，也不会按 parser/normalization policy 重算
  `known_at`。`team_match_observations`、`player_match_observations`、通用 `lineup_facts`、
  `news_evidence`、`prematch_events` 五类记录在通用 derived resolver 中仍主要只检查
  `record_id` 存在，尚无与 typed result 等价的内容身份和证据 loader；历史数据的完整重新认证
  和差异清单也未完成。
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

- 严重度：P1；状态：Open（组合与防篡改契约已实现，canonical context 和校准证据开放）。
- 当前实现证据：team-baseline v2 使用明确的中性坐标并只应用一次场地项；
  `expected-goals-composition/2` 把 baseline、context 和所有 ready lineup dimensions 以唯一贡献键
  组合进 prediction v4。不可变 calibration policy 约束公式、source path/value、reference、
  coefficient 和 bounds，所有市场视图从同一 `DixonColesGrid` 重算。
- 剩余缺口：`match-context-input/1` 只验证 derived/raw 引用与数值形状，没有从 canonical 历史
  赛程重新计算 `days_since_previous_match`；当前 policy 系数也缺少可追溯训练/校准数据集与评审
  产物，因此只能证明公式可重放，不能证明参数已经完成正式校准。
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

- 严重度：P1；状态：Open（governance evidence v2 已实现，市场与晋级生产闭环开放）。
- 当前实现证据：`EvaluationRecord/2` 和 `evaluation-comparison/2` 会从持久化 prediction、model
  run、training sample 与 typed canonical result 重算指标、固定 paired-bootstrap、reliability
  和完整 cohort 的 `all` subgroup；未来 model run、伪造概率/赛果/汇总和 cherry-picked subgroup
  均会失败。缺市场不会删除足球预测，benchmark 保持 unavailable。
- 剩余缺口：没有 authoritative market snapshot store，derived resolver 对 market refs fail closed；
  没有从已持久化 evaluation pairs 创建 comparison 的 CLI；trusted capture-run validator 和真实
  prospective cohort 不存在，也没有仓库内可证明已经评审并完成观察期的 promotion policy。
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

- 严重度：P1；状态：契约已实现，生产证据开放。
- 当前实现证据：match-report 解析在写事实前交叉校验 raw 页面、赛事/赛季、双方、日期、比分、
  URL 和来源身份；生产导入固定要求七张表，显式自定义表集合仅用于 preview/研究，不能满足
  完整报告覆盖门禁；强门禁会从 RawArchive 重放并核对 parser version、source match、主客方向、
  赛事/赛季、日期、90 分钟比分、逐队表集合、缺表和阻断诊断；缺失表、重复表、未知表及逐队
  缺表均形成结构化诊断并阻断相应事实写入。真实 380 场成功报告覆盖仍未取得。
- 设计依据：总体设计第 5.2、16、17 节；ADR-0004、ADR-0017、ADR-0022。
- 初始复现/根因：比赛报告管线信任调用方比赛 ID/比分，未与 raw 页面身份交叉核验；解析器依赖
  golden fixture 人工提供而真实 summary 可能缺少的字段。
- 修复范围：从 raw 证据解析并核对赛事、双方、日期、比分和来源 match ID；按真实 schema
  显式处理可选字段与表格变体；逐表保存解析诊断、版本和缺失原因；每场持久化采集尝试。
- 依赖顺序：依赖 R02 身份解析和 R08 raw/attempt 清单，再推进 R03 全赛季门禁和 R04 归档。
- 明确非目标：不绕过 Cloudflare/访问控制，不用 Football-Data 替代 FBref 球队/球员统计。
- 验收测试/证据：由真实保存页面裁剪的离线样本覆盖 summary、passing、defense、possession、
  misc、keeper 等约定表；身份/比分不一致失败；在线 403 返回持久诊断和退出码 3。
- 完成定义：英超 380 场每场都有 attempt；成功报告可从 raw 重放为一致 canonical 事实，
  失败报告保持 pending 且可补采，不能因 golden 解析成功宣称全季完成。

### FDP-R08：derived 血缘、运行清单和静态报告未闭环

- 严重度：P2；状态：Open（manifest 与 typed resolver 已部分实现，字节及事实闭环开放）。
- 当前实现证据：derived artifact/run manifest 校验 namespace、内容身份、时间、状态、input/output
  refs 和重复 output 的语义一致性；失败/partial 运行、command output、training/model、governance、
  paper ledger、match-report 与 official-lineup contract 已有相应持久化或 replay 路径。training 的
  通用 `derived-artifact` resolver 会递归遍历 manifest `input_refs`，只接受 succeeded/partial，
  并拒绝循环、无 authoritative `available_at` 或晚于 manifest `generated_at` 的输入；write/load
  使用同一门禁。
- 剩余缺口：`file-sha256` 目前只校验引用语法，resolver 不定位或重新哈希对应文件字节，不能
  描述为可离线取回；typed result resolver 只核对内容 ID、精确 evidence、RawArchive 注册/字节，
  并要求 fact/raw `observed_at` 一致，但没有版本化 parser/normalization contract 来从 raw 重算
  比分和 `known_at`；除 match-report 与 official-lineup 两个 parser contract 外的其余 canonical
  facts 仍缺相应 typed replay verifier。
  通用 prediction resolver 只做较浅的 hash/manifest/composition 检查，正式消费者虽已使用共享
  完整 loader，registry 本身仍需避免将浅检查描述成完整 domain replay。
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
  versioned 持久化配置；真实新闻、伤停、停赛和双方官方阵容持续 collector/健康运行未证明，
  prospective capture-run 也尚未实现。
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

- 严重度：P1；状态：Open（readiness 与 lineup-delta 契约已实现，结构化观察开放）。
- 当前实现证据：画像 readiness 校验角色必需维度、N/A、分钟/比赛阈值、窗口和 `as_of`；
  lineup delta 校验双方 11 人、非空 profile refs、契约版本、样本量、输入 raw/canonical
  引用和非空适用维度；`lineup-delta-input/3` 会逐项加载 player-profile manifest、重算增量，
  并与 `official-lineup-input/2` 绑定的 canonical official lineup facts 及其 raw refs 比对。
  球员/球队归属由 canonical lineup fact 边界校验。
  `prematch-features/1` lineup 仅作为只读 legacy artifact；`prematch-features/2` 的 ready
  lineup 只接受 `/3` 证据。缺失、N/A、insufficient-sample 与 ready 保持独立。
- 剩余缺口：canonical `player_match_observations.metrics_json` 仍是宽泛指标字典，没有版本化、
  角色化的单场 observation artifact 和逐字段来源合同；player-profile manifest 虽验证形状与引用，
  存储边界仍不能从结构化 observations 独立重算画像指标。真实全季球员覆盖也未证明。
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

- 严重度：P1；状态：Open（dataset/model 核心合同和部分 availability DAG 已实现，语义重算与
  生产运行开放）。
- 当前实现证据：content-addressed training dataset 和 model-run artifacts 保存样本、资格、capture
  mode、split、feature/label 版本、排除原因、模型字节、输出 hash 与评估 cohort。score-model
  `result-90/1` label ref 必须解析为通过内容/evidence/RawArchive 字节和 observation 时间校验的 typed
  `MatchResult90`，并精确匹配 goals 与 `label_known_at`。dataset `generated_at` 不得早于每个样本
  的 `feature_known_at`、`label_known_at` 和 captured `capture_observed_at`；成功/部分成功 dataset
  的每个输入还必须具有不晚于 `generated_at` 的 authoritative `available_at`。当前能提供 semantic
  `known_at` 的 feature ref 会与样本 `feature_known_at` 交叉校验；captured sample 的每个 feature
  ref 还必须满足实际 `available_at <= sample.as_of`。通用 `derived-artifact` 会递归复用 typed
  resolver，要求每层输入不晚于所属 manifest `generated_at`，并对循环或无权威时间 fail closed。
  model run 不得早于 dataset 生成，且 `model_run.ended_at <= prediction.generated_at`。prediction
  schema v3 仅可 audit，不能被重写或晋级为当前合同。
- 剩余缺口：并非所有 canonical/derived reference 都有 typed resolver 可提供 semantic
  `known_at`；通用 derived manifest 的递归 availability 校验不会从任意 payload 推导语义时间。
  通用 canonical facts、`file-sha256` 以及 result 的 parser/normalization 语义仍未闭环；不具备
  authoritative `available_at` 的引用会 fail closed，并不等于已经完成语义重算。trusted capture-run
  仍不存在，因此这些 training gates 不能把 reconstructed 历史升级为 prospective captured，也不能
  单独证明训练集完全无泄漏；真实生产 champion/challenger 训练、复跑和回滚证据同样缺失。
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

- 严重度：P1；状态：Open（账本与结算核心合同已实现，真实市场生产闭环开放）。
- 当前实现证据：paper ledger 以 append-only revision 保存候选/拒绝、预测、model run、market、
  风险配置、stake 和结算；支持 win/loss/push/half-win/half-loss/void 规则，从 typed canonical
  result 结算，并可重算余额与敞口。默认 `PaperBetLedger` 会绑定 CanonicalFactStore 与 RawArchive；
  带赛果的 settle、load 和 entries 均执行 typed result 内容/evidence/raw-byte/observed-at gate，
  `result=None` 只允许无需赛果的 void 路径。append/load 使用共享 persisted-prediction verifier，
  summary artifact 和 recompute run 也能修复“entry 已写但 registry 尚未写”的幂等重试。
- 剩余缺口：没有 authoritative persisted market snapshot store、真实报价 cohort 或生产纸面账本；
  result 语义也尚未由版本化 parser/normalization contract 从 raw 重算。因此当前只能证明账本领域
  合同和默认字节血缘门禁，不能证明真实赔率 ROI、运营质量或模型晋级。
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
