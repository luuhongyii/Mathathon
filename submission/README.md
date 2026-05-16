# Submission 文件夹 — 直接粘贴用

每个 `.py` 文件**整份内容就是要粘进 Mathathon 编辑器的代码**,不含任何多余文字。

## 粘贴步骤(避免缩进错乱 / IndentationError)

1. 打开要用的文件,`Ctrl+A` 全选 → `Ctrl+C` 复制
2. 平台编辑器里 `Ctrl+A` 全选 → `Delete` 删干净
3. `Ctrl+V` 粘贴
4. **检查**:第一行 `import ...` 必须顶格(无前导空格)
5. `Submit Code`

## 平台铁律(已踩坑确认)

- **绝对不要写 stderr**(`sys.stderr.write` / `print(..., file=sys.stderr)`)。
  平台把 stderr 当错误信号:一写,这一手就被判 `ERROR`、没有 `OUT`,等于弃权。
- 调试别用 stderr。要看输入格式,直接看 Gamelog 里的 `IN` 行。
- 提交前确认代码顶格无误,IndentationError 同样会判 ERROR。

## 文件说明

| 文件 | 何时用 |
|---|---|
| `rps_demo.py` | 演示局 rock-paper-scissors 的**最终版**。纯均匀随机 = 博弈论最优(看不到对手,不可被利用)。 |
| `rps_adaptive.py` | 仅当某游戏每轮输入**确实带对手历史**时才有用。RPS 演示局输入是随机笑话,用不上。留给真比赛参考。 |
| `territory_wars.py` | Territory Wars(31×31 贪吃蛇/Tron,4 人抢地盘)。对最近的对手做**迭代加深极小化极大对抗搜索**(alpha-beta + 走法排序),叶子用多源 BFS 评估 `我的Voronoi领地 - 0.6×对手领地` —— 负项让它**主动切割**对手空间;理性对手不会自杀,搜索据此忽略对手的送死走法;与所有对手隔开后切换到贴墙(Warnsdorff)填满残局。每步限时 `MOVE_BUDGET`(默认 0.07s,按平台真实限时调)。本地 1v1 打贪心 bot 18/20、4 人混战 8/10、对随机 16/16,单步 ~70ms。 |

### territory_wars.py 可调参数(文件顶部常量)
- `MOVE_BUDGET` —— 每步思考时间预算。**先看平台 Wiki 的真实限时**:若是 100ms 保持 0.07;更宽松可调大(搜得更深更强),更紧则调小。
- `AGGR` —— 切割侵略性(0=只顾自己抢地,1=完全零和对抗)。默认 0.6 偏进攻。

## 已确认事实(RPS 演示局)

- 每轮 `input()` 收到的是**随机笑话**(flavor text),不含对手出招、不含状态。
- 因此 RPS 演示局纯靠运气,均匀随机就是最优,目标只是跑通流程。

## 真比赛开始时

把新游戏的规则页发给 Claude。先跑一局看 Gamelog 的 `IN` 行确认真实输入格式,
再用 `mathathon_kit` 里的引擎(Minimax / MCTS / solve_zero_sum / RegretMatching)适配。
