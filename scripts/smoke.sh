#!/usr/bin/env bash
# ch01 三条验收标准的一键复现。用法：bash scripts/smoke.sh
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"

# 把一次 /api/chat 响应的 delta 帧拼成回复正文。验收 2 与负对照共用。
# 断言对象必须是拼接后的文本，不能是原始 SSE 流：上游按 token 切分 delta，
# 订单号会被拆成多个独立帧（见验收 2 注释）。
sse_reply() {
  printf '%s' "$1" | grep '^data: ' | sed 's/^data: //' \
    | python3 -c '
import json,sys
out=[]
for line in sys.stdin:
    d=json.loads(line)
    if "text" in d: out.append(d["text"])
print("".join(out))
'
}

echo "======================================================================"
echo "验收 1：curl 对话看到流式输出（观察 delta 是否逐条到达）"
echo "======================================================================"
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"我买的鞋子有点大，想换一双，怎么操作？"}'
echo

echo
echo "======================================================================"
echo "验收 2：连续两轮，第二轮必须能看到第一轮的上下文"
echo "======================================================================"
R1=$(curl -sN -X POST "$BASE/api/chat" -H 'Content-Type: application/json' \
  -d '{"message":"我叫张三，我的订单号是 A12345"}')
SID=$(printf '%s' "$R1" | grep -m1 '^data: ' | sed 's/^data: //' | python3 -c 'import json,sys;print(json.load(sys.stdin)["session_id"])')
echo "第一轮 session_id = $SID"

R2=$(curl -sN -X POST "$BASE/api/chat" -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"message\":\"我的订单号是多少？\"}")
echo "第二轮回复："
REPLY=$(sse_reply "$R2")
printf '%s\n' "$REPLY"
if printf '%s' "$REPLY" | grep -q 'A12345'; then
  echo "✅ 验收 2 通过：第二轮回复中出现了第一轮给出的订单号 A12345"
else
  echo "❌ 验收 2 失败：第二轮回复中未出现 A12345，上下文没有生效"
fi

echo
echo "负对照（spec §11）：新会话问同样的问题，断言必须失败"
echo "如果新会话也出现 A12345，说明上面的检查是空过的，不能证明上下文起了作用"
R3=$(curl -sN -X POST "$BASE/api/chat" -H 'Content-Type: application/json' \
  -d '{"message":"我的订单号是多少？"}')
REPLY3=$(sse_reply "$R3")
if printf '%s' "$REPLY3" | grep -q 'A12345'; then
  echo "❌ 负对照失败：新会话回复中出现了 A12345 —— 验收 2 的检查无区分力"
else
  echo "✅ 负对照通过：新会话回复中没有 A12345，验收 2 的检查确有区分力"
fi

echo
echo "======================================================================"
echo "验收 3：发一段售后描述，看到结构化 json"
echo "======================================================================"
curl -s -X POST "$BASE/api/extract" \
  -H 'Content-Type: application/json' \
  -d '{"text":"订单 A123 我要退款，希望原路退回"}' | python3 -m json.tool
