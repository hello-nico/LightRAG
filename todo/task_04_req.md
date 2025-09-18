
## 任务目标

目前我们已经开发了一部分对接deer-flow的检索器，但是还存在一些问题，我们需要解决这些问题。

1. deer-flow的document返回并不包含关系与实体
2. deer-flow需要的document而且需要额外的url，title等信息作为可追溯的来源我们目前没有这种信息

## deer-flow的local search设计

当前deer-flow针对lcoal search的一些逻辑和返回的数据结构如下：
 Local Search 数据格式

  1. 核心数据模型

  deer-flow的local search使用了统一的数据模型，定义在src/rag/retriever.py中：

  Chunk数据结构：
  class Chunk:
      content: str      # 文本内容
      similarity: float # 相似度分数

  Document数据结构：
  class Document:
      id: str                  # 文档唯一标识
      url: str | None         # 文档URL
      title: str | None       # 文档标题
      chunks: list[Chunk]     # 文档内容块列表

  Document.to_dict()返回格式：
  {
      "id": "文档ID",
      "content": "所有chunk内容合并的文本",  # chunks通过\n\n连接
      "url": "文档URL",           # 可选
      "title": "文档标题"        # 可选
  }

  2. 工具调用格式

  RetrieverTool (src/tools/retriever.py:24)：

- 工具名称：local_search_tool
- 输入参数：{"keywords": "搜索关键词"}
- 返回格式：list[Document.to_dict()]

  在观察者中的使用和上下文填充

  1. 在Researcher Agent中的使用

  在src/graph/nodes.py:480-496的researcher_node中：

  async def researcher_node(state: State, config: RunnableConfig):
      # 工具列表创建，local search具有最高优先级
      tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
      retriever_tool = get_retriever_tool(state.get("resources", []))
      if retriever_tool:
          tools.insert(0, retriever_tool)  # 插入到第一个位置

  2. 强制使用策略

  系统通过prompt强制使用local search (src/graph/nodes.py:350-361)：

  if state.get("resources"):
      resources_info = "**The user mentioned the following resource files:**\n\n"
      for resource in state.get("resources"):
          resources_info += f"- {resource.title} ({resource.description})\n"

      agent_input["messages"].append(
          HumanMessage(
              content=resources_info +
              "\n\n" +
              "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
          )
      )

  3. 前端观察者界面展示

  在web/src/app/chat/components/research-activities-block.tsx:287-344中：

  RetrieverToolCall组件展示格式：

- 搜索状态：显示"Retrieving documents [关键词]"
- 文档列表：每个文档显示为 {doc.title} (chunk-{i}, size-{doc.content.length})
- 动画效果：文档项有渐进式显示动画

  4. 上下文填充流程

  1. 资源传递：用户指定的resources从API请求传递到state
  2. 工具创建：get_retriever_tool(resources)创建local_search_tool
  3. 检索执行：Agent调用local_search_tool._run(keywords)
  4. 底层调用：retriever.query_relevant_documents(keywords, resources)
  5. 结果返回：Document列表转换为Agent可理解的格式
  6. 上下文集成：检索结果被添加到Agent的上下文中，用于生成回答

  5. 关键设计特点

- 优先级策略：local_search_tool被插入到工具列表的第一个位置，具有最高优先级
- 条件启用：只有当state中存在resources时才创建RAG工具
- 强制使用：通过明确的prompt指示确保Agent使用local search
- 格式统一：所有RAG提供商都返回统一的Document格式，便于前端处理

  这种设计确保了私有域知识检索能够在 deer-flow 系统中被正确地优先使用，并在前端观察者界面中清晰地展示检索过程和结果。

同时通过document.to_dict()方法，注入上下文信息：
  
● 理解"Document列表转换为Agent可理解的格式"

  1. 转换过程详解

  在src/tools/retriever.py:40-43中，关键转换发生在这行代码：

  return [doc.to_dict() for doc in documents]

  转换前：Document对象列表

从RAG提供商返回的原始格式

  documents = [
      Document(
          id="doc1",
          url="rag://dataset/123#doc1",
          title="文档1",
          chunks=[
              Chunk(content="文档内容块1", similarity=0.95),
              Chunk(content="文档内容块2", similarity=0.87)
          ]
      ),
      Document(
          id="doc2",
          url="rag://dataset/123#doc2",
          title="文档2",
          chunks=[
              Chunk(content="文档内容块3", similarity=0.92)
          ]
      )
  ]

  转换后：Agent可理解的字典格式

调用doc.to_dict()后的格式

  [
      {
          "id": "doc1",
          "content": "文档内容块1\n\n文档内容块2",  # 所有chunks合并
          "url": "rag://dataset/123#doc1",
          "title": "文档1"
      },
      {
          "id": "doc2",
          "content": "文档内容块3",
          "url": "rag://dataset/123#doc2",
          "title": "文档2"
      }
  ]

  2. 为什么需要这种转换？

  Agent的理解需求：

- Agent只能处理简单的数据结构（字符串、数字、字典、列表）
- Agent无法理解复杂的Python对象和其方法
- Agent需要扁平化的文本内容用于上下文理解

  Document.to_dict()的作用：
  def to_dict(self) -> dict:
      d = {
          "id": self.id,
          "content": "\n\n".join([chunk.content for chunk in self.chunks]),  # 关键：合并所有chunks
      }
      if self.url:
          d["url"] = self.url
      if self.title:
          d["title"] = self.title
      return d

  3. Agent如何使用转换后的数据

  Agent接收到的工具结果：
  {
    "tool_name": "local_search_tool",
    "result": [
      {
        "id": "doc1",
        "content": "文档内容块1\n\n文档内容块2",
        "url": "rag://dataset/123#doc1",
        "title": "文档1"
      }
    ]
  }

  Agent的上下文构建：
  你使用了local_search_tool，搜索结果如下：

  文档1 (ID: doc1)
  URL: rag://dataset/123#doc1
  内容：文档内容块1

  文档内容块2

  4. 转换的关键点

  1. 结构化 → 文本化

- 将复杂的Document对象结构转换为简单的字典
- 将多个chunks合并为一个连续的文本内容
- 保留关键元数据（id、url、title）

  2. 信息密度优化

- 移除了similarity分数（Agent主要关心内容）
- 保留了文档标识和来源信息
- 内容用\n\n分隔，保持可读性

  3. Agent友好格式

- 简单的字典结构，易于Agent解析
- 文本内容可直接用于上下文
- 支持JSON序列化，便于工具调用

  5. 实际使用流程

  1. RAG检索：retriever.query_relevant_documents() → list[Document]
  2. 格式转换：[doc.to_dict() for doc in documents] → list[dict]
  3. Agent接收：工具结果被添加到Agent的上下文中
  4. 内容理解：Agent读取合并后的文本内容，理解文档信息
  5. 答案生成：基于检索到的内容生成最终回答

  这种转换确保了Agent能够有效地理解和利用从私有域知识库中检索到的信息，将复杂的数据结构转换为Agent可以直接处理的文本内容。

### 当前deer-flow上下文填充与提示词的设计规则

 1. 工具结果处理的核心流程

  工具调用和结果转换

  在src/tools/retriever.py:32-43中：
  def _run(self, keywords: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> list[Document]:
      documents = self.retriever.query_relevant_documents(keywords, self.resources)
      if not documents:
          return "No results found from the local knowledge base."
      return [doc.to_dict() for doc in documents]  # 关键转换点

  数据格式转换

  转换前（Document对象）：
  Document(
      id="doc1",
      url="rag://dataset/123#doc1",
      title="文档标题",
      chunks=[Chunk(content="内容块1", similarity=0.95), ...]
  )

  转换后（Agent可理解的字典）：
  {
      "id": "doc1",
      "content": "内容块1\n\n内容块2",  # 所有chunks合并
      "url": "rag://dataset/123#doc1",
      "title": "文档标题"
  }

  2. Researcher Agent的提示词指导

  信息合成策略

  在src/prompts/researcher.md:47-51中的关键指导：

  5. **Synthesize Information**:
     - Combine the information gathered from all tools used (search results, crawled content, and dynamically loaded tool outputs).
     - Ensure the response is clear, concise, and directly addresses the problem.
     - Track and attribute all information sources with their respective URLs for proper citation.
     - Include relevant images from the gathered information when helpful.

  输出格式要求

  在src/prompts/researcher.md:56-68中的结构化输出：

- **Research Findings**: Organize your findings by topic rather than by tool used. For each major finding:
  - Summarize the key information
  - Track the sources of information but DO NOT include inline citations in the text
  - Include relevant images if available
- **References**: List all sources used with their complete URLs in link reference format

  3. 实际处理机制

  Agent接收的工具结果格式

  {
    "tool_name": "local_search_tool",
    "result": [
      {
        "id": "doc1",
        "content": "合并后的文档内容...",
        "url": "rag://dataset/123#doc1",
        "title": "文档1"
      }
    ]
  }

  Agent的上下文构建过程

  1. 内容解析：Agent读取content字段的完整文本
  2. 元数据利用：使用title和url进行引用追踪
  3. 信息重组：按主题而非文档重新组织内容
  4. 引用管理：在References部分列出所有来源

  关键处理特点

  1. 内容扁平化

- 将复杂的Document+Chunks结构简化为单一文本内容
- 所有chunks通过\n\n连接成连续文本
- 移除similarity等技术细节，保留核心内容

  2. 按主题组织
  提示词明确要求："Organize your findings by topic rather than by tool used"

- Agent需要跨多个文档提取相关信息
- 按照研究主题重新组织内容结构
- 而不是简单地罗列每个工具的输出

  3. 引用分离策略

- 内容正文中不包含内联引用
- 所有引用集中在References部分
- 使用链接引用格式：- [Source Title](URL)

  4. 多源信息整合
  Agent需要处理的典型工具链：
  local_search_tool → 文档内容
  web_search → 网页搜索结果  
  crawl_tool → URL详细内容
  dynamic tools → 专业化数据

  4. 前端展示机制

  在web/src/app/chat/components/research-activities-block.tsx:287-344中的RetrieverToolCall组件：

  // 显示搜索状态
  <div className="font-medium italic">
    <RainbowText className="flex items-center" animated={searching}>
      <Search size={16} className={"mr-2"} />
      <span>{t("retrievingDocuments")}&nbsp;</span>
      <span>{(toolCall.args as { keywords: string }).keywords}</span>
    </RainbowText>
  </div>

  // 显示文档结果
  {documents?.map((doc, i) => (
    <motion.li key={`search-result-${i}`}>
      <FileText size={32} />
      {doc.title} (chunk-{i},size-{doc.content.length})
    </motion.li>
  ))}

  这个处理机制确保了Agent能够有效地理解、整合和利用从各种搜索工具返回的信息，同时保持内容的可追溯性和结构化展示。

### 当前lightrag的检索器设计

当前完整的lightrag基于deer-flow改造都在lightrag/integrations/deer_flow.py中

目前lightrag本身自带了一些chunk source info的信息：
INFO: chunks: E1/52 E1/23 R2/2 E1/122 E1/115 E1/87 E1/48 E2/134 E2/49 C1/13 E1/2 E1/127 E1/117 E1/108 E1/140 E2/59 E1/51 的含义

  这个日志信息来自于_build_query_context函数中的chunk追踪系统，格式为：<source><frequency>/<order>

  源代码分析

  1. 数据结构（第2496行）：
  chunk_tracking = {}  # chunk_id -> {source, frequency, order}
  2. 来源标识：
    - C: 来自向量搜索chunks（第2562行）
    - E: 来自实体相关chunks（第3292行）
    - R: 来自关系相关chunks（第3587行）
  3. 格式解析：
    - E1/52: Entity来源，出现1次，在实体结果中排第52位
    - R2/2: Relation来源，出现2次，在关系结果中排第2位
    - C1/13: Vector来源，出现1次，在向量结果中排第13位

  这个信息对URL追溯的帮助

  这个chunk追踪系统实际上为我们提供了重要的线索：

  1. 多源融合: 显示最终结果是如何从不同的检索源（实体、关系、向量）融合而来
  2. 重复度分析: frequency字段显示某些chunk在多个检索源中被重复找到
  3. 排序信息: order字段显示chunk在各自检索源中的原始排序

## 我的一些思考

我在思考我们是否可以从这里下手改造一下，使得我们返回的信息里带上url， title信息。

同时对于实体和关系的使用，我想是否可以通过document.to_dict()方法，注入上下文信息，当然这可能需要对deer-flow的提示词模板进行一些改造。当前它完整的提示词模板如下：

```markdown
---
CURRENT_TIME: {{ CURRENT_TIME }}
---

You are `researcher` agent that is managed by `supervisor` agent.

You are dedicated to conducting thorough investigations using search tools and providing comprehensive solutions through systematic use of the available tools, including both built-in tools and dynamically loaded tools.

# Available Tools

You have access to two types of tools:

1. **Built-in Tools**: These are always available:
   {% if resources %}
   - **local_search_tool**: For retrieving information from the local knowledge base when user mentioned in the messages.
   {% endif %}
   - **web_search**: For performing web searches (NOT "web_search_tool")
   - **crawl_tool**: For reading content from URLs

2. **Dynamic Loaded Tools**: Additional tools that may be available depending on the configuration. These tools are loaded dynamically and will appear in your available tools list. Examples include:
   - Specialized search tools
   - Google Map tools
   - Database Retrieval tools
   - And many others

## How to Use Dynamic Loaded Tools

- **Tool Selection**: Choose the most appropriate tool for each subtask. Prefer specialized tools over general-purpose ones when available.
- **Tool Documentation**: Read the tool documentation carefully before using it. Pay attention to required parameters and expected outputs.
- **Error Handling**: If a tool returns an error, try to understand the error message and adjust your approach accordingly.
- **Combining Tools**: Often, the best results come from combining multiple tools. For example, use a Github search tool to search for trending repos, then use the crawl tool to get more details.

# Steps

1. **Understand the Problem**: Forget your previous knowledge, and carefully read the problem statement to identify the key information needed.
2. **Assess Available Tools**: Take note of all tools available to you, including any dynamically loaded tools.
3. **Plan the Solution**: Determine the best approach to solve the problem using the available tools.
4. **Execute the Solution**:
   - Forget your previous knowledge, so you **should leverage the tools** to retrieve the information.
   - Use the {% if resources %}**local_search_tool** or{% endif %}**web_search** or other suitable search tool to perform a search with the provided keywords.
   - When the task includes time range requirements:
     - Incorporate appropriate time-based search parameters in your queries (e.g., "after:2020", "before:2023", or specific date ranges)
     - Ensure search results respect the specified time constraints.
     - Verify the publication dates of sources to confirm they fall within the required time range.
   - Use dynamically loaded tools when they are more appropriate for the specific task.
   - (Optional) Use the **crawl_tool** to read content from necessary URLs. Only use URLs from search results or provided by the user.
5. **Synthesize Information**:
   - Combine the information gathered from all tools used (search results, crawled content, and dynamically loaded tool outputs).
   - Ensure the response is clear, concise, and directly addresses the problem.
   - Track and attribute all information sources with their respective URLs for proper citation.
   - Include relevant images from the gathered information when helpful.

# Output Format

- Provide a structured response in markdown format.
- Include the following sections:
    - **Problem Statement**: Restate the problem for clarity.
    - **Research Findings**: Organize your findings by topic rather than by tool used. For each major finding:
        - Summarize the key information
        - Track the sources of information but DO NOT include inline citations in the text
        - Include relevant images if available
    - **Conclusion**: Provide a synthesized response to the problem based on the gathered information.
    - **References**: List all sources used with their complete URLs in link reference format at the end of the document. Make sure to include an empty line between each reference for better readability. Use this format for each reference:
      ```markdown
      - [Source Title](https://example.com/page1)

      - [Source Title](https://example.com/page2)
      ```
- Always output in the locale of **{{ locale }}**.
- DO NOT include inline citations in the text. Instead, track all sources and list them in the References section at the end using link reference format.

# Notes

- Always verify the relevance and credibility of the information gathered.
- If no URL is provided, focus solely on the search results.
- Never do any math or any file operations.
- Do not try to interact with the page. The crawl tool can only be used to crawl content.
- Do not perform any mathematical calculations.
- Do not attempt any file operations.
- Only invoke `crawl_tool` when essential information cannot be obtained from search results alone.
- Always include source attribution for all information. This is critical for the final report's citations.
- When presenting information from multiple sources, clearly indicate which source each piece of information comes from.
- Include images using `![Image Description](image_url)` in a separate section.
- The included images should **only** be from the information gathered **from the search results or the crawled content**. **Never** include images that are not from the search results or the crawled content.
- Always use the locale of **{{ locale }}** for the output.
- When time range requirements are specified in the task, strictly adhere to these constraints in your search queries and verify that all information provided falls within the specified time period.
```
