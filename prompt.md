## 目标
我要使用Superpowers模式做一个电商智能客服系统,第一步先纯跑通对话
## 功能需求
1.对话接口:支持多轮对话,实现SSE流式输出,逐token推送
2.Prompt管理:使用PromptTemplate模板化,用System Prompt定义客服角色设定和行为约束
3.结构化输出:把用户的售后需求描述提取成订单号,诉求期望,期望方案这样的固定字段用with_structured_output 实现.
4.多轮上下文管理先做最简版:消息剪裁和token预算控制

## 技术栈
- FastAPI+Python+Langchain
- 模型接入直连上游,应用侧统一使用OpenAI协议,deepseek,GPT,Kimi之类的模型都能换着接

## 暂时不做
- 工具调用,Agent loop,我们先暂时跑通纯对话

## 验收标准
- curl对话可以看到流式输出
- 连续问两轮,第二轮能看到第一轮的上下文
- 发一段售后描述,能看到结构化json

## 工作要求
1. 全程走 Superpowers 流程,技能自动触发;产出不是可单测代码的任务(纯 Prompt、数据类),把 TDD 那步换成拿标注样例或评估集跑一遍验证,其余步骤照走;聊天页面是例外,用 Vibe Coding 方式直接做,我描述效果你改,不套 brainstorm、TDD、code review 那套流程
2. 过程留痕:在仓库 dev-notes/ch01.md 里追记开发过程,每完成一个阶段(brainstorm 定稿、计划评审通过、每个任务完成、code review 结论、finish)就补一段,记四样:我这一步发的关键原话、你的关键产出(spec / plan 路径、评审结论)、我拒绝或纠偏了什么、翻车与返工;不许收尾时一次性补记
3. 涉及具体库、框架、API 的用法(FastAPI、SQLAlchemy、LangChain、LangGraph、Milvus、Langfuse 这些),一律先用 Context7 MCP 查最新官方文档和接口定义再动手,别凭记忆写,版本对不上的 API 是返工重灾区
4. 上面点名的技术选型是定死的,实现中发现矛盾或走不通,停下来问我,不要自行换方案
5. 完结交付:功能演示命令、测试结果、dev-notes 路径