# main.py
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
# 假设你已经有了封装好的内部大模型调用接口
# from core.llm_config import invoke_internal_llm 

from prompts.rca_prompts import RCA_SYSTEM_PROMPT

# 1. 定义状态 (State) - 贯穿整个工作流的数据结构
class GraphState(TypedDict):
    log_path: str
    extracted_error_text: str
    llm_summary: str
    human_feedback: str

# 2. 定义节点 (Nodes)
def extract_log_node(state: GraphState):
    """节点 1：读取文件并提取关键报错信息（这里用 Mock 数据代替复杂的正则提取）"""
    print(">>> [Node: Log Extractor] 正在扫描指定目录，提取最近的失败日志...")
    # 实际开发中，这里应该是: with open(state['log_path']) as f: ...
    # 为了 MVP 演示，我们硬编码一段典型的报错日志
    mock_log_snippet = """
    [INFO] 2026-03-03 10:00:01 - Starting model conversion pipeline...
    [INFO] 2026-03-03 10:00:15 - Loading weights from /models/resnet50.onnx
    [ERROR] 2026-03-03 10:00:18 - Conversion failed at layer 'Conv_0'. 
    Traceback (most recent call last):
      File "convertor.py", line 142, in process_layer
        assert input_shape == weight_shape, "Shape mismatch detected."
    AssertionError: Shape mismatch detected. Expected (1, 3, 224, 224), got (1, 4, 224, 224).
    """
    return {"extracted_error_text": mock_log_snippet}

def llm_analysis_node(state: GraphState):
    """节点 2：调用大模型进行归因分析"""
    print(">>> [Node: LLM Analyzer] 正在调用 gpt-oss-120b 进行深度分析...")
    
    # 组装 Message。注意：这里只传基础文本，不使用复杂的 tool calls 或 kwargs，确保内部模型兼容
    messages = [
        SystemMessage(content=RCA_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the log snippet:\n{state['extracted_error_text']}")
    ]
    
    # 实际开发中调用你的模型: 
    # response = invoke_internal_llm(messages)
    # llm_output = response.content
    
    # Mock LLM 返回结果
    llm_output = """
    1. Error Type: AssertionError (Shape Mismatch)
    2. Failing Component/Layer: convertor.py, line 142 (layer 'Conv_0')
    3. Suspected Root Cause: The input tensor has 4 channels (e.g., RGBA) instead of the expected 3 channels (RGB) for the ResNet50 model conversion.
    4. Recommended Action: Add a preprocessing step to drop the alpha channel (slice input[:, :3, :, :]) before feeding it to the conversion pipeline.
    """
    print(f"\n--- LLM 分析报告 ---\n{llm_output}\n--------------------\n")
    return {"llm_summary": llm_output}

def human_review_node(state: GraphState):
    """节点 3：人类拦截点后的处理"""
    # 这个节点只有在人类解除拦截后才会执行
    print(f">>> [Node: Human Review] 接收到人类反馈指令: {state.get('human_feedback', 'Approve')}")
    # 后续可以在这里加入将报告发送到飞书或写入本地汇总文件的逻辑
    return state

# 3. 构建状态机拓扑
workflow = StateGraph(GraphState)

workflow.add_node("Extractor", extract_log_node)
workflow.add_node("Analyzer", llm_analysis_node)
workflow.add_node("Human_Review", human_review_node)

workflow.add_edge(START, "Extractor")
workflow.add_edge("Extractor", "Analyzer")
workflow.add_edge("Analyzer", "Human_Review")
workflow.add_edge("Human_Review", END)

# 核心：设置拦截器 (MemorySaver 是必须的，用于保存中断时的状态)
memory = MemorySaver()
# 明确指定在 Human_Review 节点之前硬中断
app = workflow.compile(checkpointer=memory, interrupt_before=["Human_Review"])

# 4. 运行与交互逻辑 (CLI)
if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "test_run_001"}}
    
    # 第一阶段运行：它会自动在 Human_Review 前停下
    initial_state = {"log_path": "/var/logs/test_pipeline_latest.log"}
    print("========== 启动 Autotest-Agent ==========")
    for event in app.stream(initial_state, config=thread_config):
        pass # stream 会打印节点内部的 print 信息
    
    # 检查状态机是否处于挂起（拦截）状态
    current_state = app.get_state(thread_config)
    if current_state.next == ('Human_Review',):
        print("\n[ALERT] 流程已暂停！Agent 需要您的审阅。")
        user_input = input("请输入您的决策 (输入 'y' 接受归因并继续，或输入修改意见): ")
        
        # 将人类意见更新到状态中
        app.update_state(thread_config, {"human_feedback": user_input})
        
        # 第二阶段运行：恢复执行
        print("\n========== 恢复工作流 ==========")
        for event in app.stream(None, config=thread_config):
            pass
            
    print("\n========== 任务结束，已生成汇总信息 ==========")

# prompts/rca_prompts.py

RCA_SYSTEM_PROMPT = """You are an Elite QA Architect specializing in Root Cause Analysis (RCA) for automated testing, SDK integration, and AI model conversion pipelines. 
Your objective is to analyze truncated error logs and provide a surgically precise, actionable diagnosis.

ANALYZE THE FOLLOWING LOG SNIPPET AND EXTRACT:
1. Error Type: (e.g., Segmentation Fault, Timeout, Shape Mismatch, Dependency Missing)
2. Failing Component/Layer: (The specific file, function, or neural network layer where the crash occurred)
3. Suspected Root Cause: (A brief, highly logical explanation of WHY it failed)
4. Recommended Action: (1-2 actionable steps for the developer to fix or investigate further)

CONSTRAINT: 
- Output strictly in the above numbered list format. 
- Do not include pleasantries. Do not guess wildly. If the log snippet lacks sufficient context, explicitly state "INSUFFICIENT CONTEXT".
"""
