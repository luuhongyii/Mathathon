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
| `territory_wars.cpp` / `.exe` | **Territory Wars 的提交版 —— 用这个。** C++ bot:迭代加深极小化极大搜索(alpha-beta)+ **chamber/割点分析**评估。叶子不再数「可达格子总数」(会被自己砌的墙骗、把自己困死),而是用 Tarjan 割点把 Voronoi 领地拆成腔室,算「进了细颈之后真正能走到的可用空间」。生存闸门排除撞车方向;隔开后贴墙填残局;每步思考时间=剩余预算÷剩余回合,整局算力自动卡在 ~0.34s。本地对随机 30/30;4 人混战打 3 个贪心 bot 拿 37% 第一(均势基线 25%),1v1 打贪心 12/16。 |
| `territory_wars.py` | 较早的 Python 快速 1-ply 贪心版(无搜索),较弱但整局算力 ~130ms 也能跑。只作备份/参考,正式提交用 C++ 版。 |
| `capture_the_flag.py` | **Capture the Flag 的提交版 —— 用这个。** 29x29、2v2 夺旗。一个进程一名玩家,队友跑同一份代码(克隆);无通信,用「位置字典序」分工——一人进攻夺敌旗、一人防守守己旗。移动 = 贪心下降 BFS 距离场(敌旗/己旗/己方领地/绿洲四张静态场开局算一次,每回合最多再跑一次 BFS 追人)。危险模型:身处敌方领地且与活着的敌人切比雪夫距离 ≤2 的格子会被抓,避开;卡住太久转「拼一把」强推。绿洲回血(140)兼安全区。本地对随机 60/60 全胜,100 局零 DQ。纯 Python 足够快(整局算力 ~0.15s)。 |

### Territory Wars 提交方式(C++)

**提交 `territory_wars.cpp` 源文件本身,不要提交 .exe。**

⚠️ 已踩坑:判题机是 **Linux**。上传 Windows 编译的 `.exe` 会报
`ERROR Invalid Move(OUT): execl: Exec format error`(Linux 跑不了 Windows PE
二进制)。平台会在它自己的 Linux 机器上编译你提交的源文件。

用平台的 **`Submit C++/Binary`** 按钮上传 **`territory_wars.cpp`**(源文件)。
源码只用标准头(`<cstdio> <cstdlib> <chrono>`),无任何 Windows 依赖,已用
`g++ -std=c++17 -O2 -Wall -Wextra` 验证零警告,Linux g++ 能直接编过。

本地自测编译(仅用于测试,不是提交物):

```bash
g++ -std=c++17 -O2 territory_wars.cpp -o territory_wars      # Linux/WSL
g++ -std=c++17 -O2 -static territory_wars.cpp -o tw.exe      # 本地 Windows
```

- 顶部常量 `TOTAL_BUDGET`(默认 0.34s):整局算力上限。每步用「剩余预算÷剩余
  回合」自动分配思考时间,512 回合加起来不会超。平台限时 500ms,留 ~160ms 余量。
- `MAX_DEPTH`(默认 12)搜索深度上限;`COLLIDE_W`(默认 600)撞车惩罚。
- ⚠️ 教训:Python 对抗搜索版每步固定烧 70ms,第 7 回合累计超 500ms 被判
  `Timeouted`。**这个游戏限时是「整局总和」。** C++ 版用「预算分摊到每步」解决。

### Capture the Flag 提交方式

提交 `capture_the_flag.py` 整份内容(Python)。判题机是 Linux,Python 跑得动;
本 bot 不做搜索,开局算 4 张 BFS 场、每回合最多再 1 张,整局算力 ~0.15s,无超时风险。
本地模拟器 `tools/ctf_sim.py` 跑真实规则验证(`python tools/ctf_sim.py` 对随机、
`python tools/ctf_sim.py mirror` 自对弈)。⚠️ 测试棋盘必须**保证四角连通**——纯随机
障碍会把角落封死造出无解局;真平台棋盘密度高(~25%)但是结构化、连通的。

## 已确认事实(RPS 演示局)

- 每轮 `input()` 收到的是**随机笑话**(flavor text),不含对手出招、不含状态。
- 因此 RPS 演示局纯靠运气,均匀随机就是最优,目标只是跑通流程。

## 真比赛开始时

把新游戏的规则页发给 Claude。先跑一局看 Gamelog 的 `IN` 行确认真实输入格式,
再用 `mathathon_kit` 里的引擎(Minimax / MCTS / solve_zero_sum / RegretMatching)适配。
