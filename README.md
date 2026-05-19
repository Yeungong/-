import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# ==========================================
# 1. 定义数据结构 (Data Schema)
# ==========================================

class RawMessage(BaseModel):
    """原始摄入的消息结构"""
    source: str      # 来源：如 "Email", "IRC", "Slack"
    sender: str      # 发送人
    content: str     # 消息正文

class TaskDecision(BaseModel):
    """决策 Agent 的长链推理与结构化输出"""
    is_actionable: bool = Field(..., description="该消息是否包含需要处理的实际任务或待办事项")
    reasoning: str = Field(..., description="长链推理：分析该消息的紧急程度、潜在依赖以及为什么做出此决策")
    task_title: Optional[str] = Field(None, description="提取出的任务简短标题（如为待办）")
    priority: Optional[str] = Field(None, description="任务优先级: HIGH, MEDIUM, LOW")
    reply_draft: Optional[str] = Field(None, description="针对该消息的自动化回复草稿，若无需回复则为空")

# ==========================================
# 2. 核心 Agent 编排器 (Orchestrator)
# ==========================================

class TaskAutomationAgent:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        # 通用 OpenAI 兼容接口，可自由切换至 DeepSeek, Moonshot, Gemini 等
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-mini" # 或其他支持 JSON Mode / Structured Outputs 的模型

    def ingest_messages(self) -> List[RawMessage]:
        """
        【信息摄入层】
        此处用 Mock 数据模拟从邮件、IRC 抓取到的碎片化信息
        """
        print("[Ingestion] 正在摄入未读消息...")
        return [
            RawMessage(
                source="IRC", 
                sender="Linux-Dev-Bot", 
                content="⚠️ 警告：服务器 10.0.0.5 内存占用超过 92%，可能引发 OOM，请密切关注。"
            ),
            RawMessage(
                source="Email", 
                sender="manager@company.com", 
                content="下周二下午两点我们需要开会讨论一下 Q2 的技术架构演进，收到请回复。"
            ),
            RawMessage(
                source="Slack", 
                sender="Colleague_Zhang", 
                content="哈哈，今晚烤肉去吗？听说南门新开了一家店，味道不错！"
            )
        ]

    def reset_reasoning_and_decide(self, message: RawMessage) -> TaskDecision:
        """
        【决策与长链推理层】
        调用大模型，强制要求其进行长链思考，并返回结构化 JSON
        """
        print(f"\n[Reasoning] 决策 Agent 正在分析来自 [{message.source}] 的消息...")
        
        system_prompt = (
            "你是一个高效的个人数字化助理。你的任务是分析用户接收到的碎片化消息。\n"
            "你必须严格按照要求的 JSON 结构进行回复。\n"
            "在 reasoning 字段中，请展现你的长链推理过程，包括：\n"
            "1. 评估消息的真实意图和紧迫性。\n"
            "2. 判断是否需要用户介入或记录为待办。\n"
            "3. 决定是否需要生成自动化回复。"
        )
        
        user_prompt = f"来源: {message.source}\n发送人: {message.sender}\n内容: {message.content}"

        # 使用 Beta 阶段的结构化输出 API (或使用 Response Format JSON)
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=TaskDecision,
        )
        
        return completion.choices[0].message.parsed

    def execute_actions(self, message: RawMessage, decision: TaskDecision):
        """
        【执行层】
        根据决策 Agent 的结果，自动编排并调用相应的底层工具链
        """
        print(f"[Execution] 开启执行管道:")
        print(f" ├─ 推理链路: {decision.reasoning}")
        
        if decision.is_actionable:
            print(f" ├─ 🚨 [触发自动化工具1：待办创建] 成功向待办系统推送任务!")
            print(f" │    📌 任务: {decision.task_title} | 优先级: {decision.priority}")
        else:
            print(f" ├─ ℹ️  [跳过待办创建] 评估为非任务类型信息。")

        if decision.reply_draft:
            print(f" └─ ✉️  [触发自动化工具2：自动回复] 已生成草稿至 {message.source} 发送队列:")
            print(f"      「 {decision.reply_draft} 」")
        else:
            print(f" └─ ℹ️  [跳过自动回复] 该消息无需或不宜进行自动回复。")

    def run_pipeline(self):
        """启动整个 Agent 自动化编排流"""
        messages = self.get_mock_messages() if hasattr(self, 'get_mock_messages') else self.ingest_messages()
        
        print(f"=== 成功捕获 {len(messages)} 条原始数据，启动 Agent 编排流水线 ===")
        for msg in messages:
            # 1. 决策
            decision = self.reset_reasoning_and_decide(msg)
            # 2. 执行
            self.execute_actions(msg, decision)
        print("\n=== 所有任务处理完毕，流水线安全闭环 ===")

# ==========================================
# 3. 运行入口
# ==========================================
if __name__ == "__main__":
    # 请替换为你的实际 API Key 和 Base URL
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if API_KEY == "your-api-key-here":
        print("❌ 请先配置您的 API_KEY 才能运行该 Agent 实例。")
    else:
        agent = TaskAutomationAgent(api_key=API_KEY, base_url=BASE_URL)
        agent.run_pipeline()
