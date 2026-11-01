import logging, re, json
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Dict, Tuple, List, Any
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE,ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from knowledge.processor.query_processor.base import get_config
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class ItemNameExtractor:
    """调用 LLM 从用户查询 + 历史中提取商品名称。"""

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        # 1. 获取 LLM 客户端（开启 JSON 模式）
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except (EnvironmentError, ConnectionError) as e:
            logger.error(f"LLM客户端获取失败: {e}")
            return result

        # 2. 组装提示词
        human_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_text if history_text else "暂无上下文",
            query=original_query
        )

        # 3. 调用 LLM + 解析
        llm_response = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        llm_content = llm_response.content.strip()

        if not llm_content:
            return result
        try:
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")
        except Exception as e:
            logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        return result

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        # 1. 清洗 Markdown 代码块围栏（防御性保留，即使设置了 response_format）
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. JSON 解析
        parsed_llm_result: Dict[str, Any] = json.loads(content)

        # 3. 清洗 item_names（过滤空值、非字符串元素，并去除前后空格）
        raw_item_names = parsed_llm_result.get('item_names')
        if not isinstance(raw_item_names, list):
            clean_item_names = []
        else:
            clean_item_names = [
                str(raw_item).strip()
                for raw_item in raw_item_names
                if isinstance(raw_item, str) and raw_item.strip()
            ]

        # 4. 清洗 rewritten_query
        raw_rewritten_query = parsed_llm_result.get('rewritten_query')
        clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

        return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}


class ItemNameAligner:
    def __init__(self):
        self._config = get_config()

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        检索向量数据库并且和向量数据库中的商品名对齐 最终返回确定的商品名列表或者模糊的商品名列表
        Args:
            item_names: 商品名称列表。

        Returns:

        """
        # 1. 混合检索向量数据库
        try:
            search_result = self._search_vector(item_names)
        except Exception as e:
            logger.error(f"向量匹配异常: {e}")
            return [], []

        # 2. 根据混合向量检索到结果做对齐【confirmed/options 】
        confirmed, options = self._align(search_result)

        # 3. 分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        # 4. 返回确定的confirmed容器和options容器
        return confirmed, options

    def _search_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
         对LLM提取到的所有商品名进行向量检索
        Args:
            item_names: LLM提取到商品名列表

        Returns:
          Dict[str, Any]:
          例子：{"extracted_name":"LLM提取的商品名1","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}
          例子：{"extracted_name":"LLM提取的商品名2","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}

        """
        final_search_result = []
        # 1. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return final_search_result
        # 2. 获取BGE-M3嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"BGE-M3客户端获取失败 原因:{str(e)}")
            return final_search_result

        # 3. 商品名列表向量化(混合向量)
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_m3_client, embedding_documents=item_names)
        except Exception as e:
            logger.error(f"商品列表{item_names}生成混合向量失败 原因:{str(e)} ")
            return final_search_result

        # 4. 混合向量检索
        for index, item_name in enumerate(item_names):
            # 4.1 构建稠密以及混合向量的检索请求
            hybrid_requests = create_hybrid_search_requests(hybrid_vector_result['dense'][index],
                                                            hybrid_vector_result['sparse'][index])

            # 4.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(milvus_client=milvus_client,
                                                               collection_name=self._config.item_name_collection,
                                                               search_requests=hybrid_requests,
                                                               ranker_weights=(0.5, 0.5),
                                                               norm_score=True,
                                                               limit=5,
                                                               output_fields=['item_name']
                                                               )

            # 4.3 解析结果

            matches = [{"score": item_search_res['distance'], "item_name": item_search_res['entity']['item_name']} for
                       item_search_res in
                       (hybrid_search_result[0] if hybrid_search_result else [])]

            # 4.4 封装
            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        主要职责：对齐
        怎么对齐？
        将什么样的商品名item_name放到confirmed
        将什么样的商品名item_name放到options
        将什么样的商品名item_name两个容器都不放
        制定规则：
        两个规则：1.如果从向量数据库中查到的商品名分数比如大于0.7 放到confirmed
        两个规则：2.如果从向量数据库中查到的商品名分数比如小于等于0.7大于0.45 放到options
                 3.如果从向量数据库中查到的商品名分数小于等于0.45 两个容器都不放
        疑问：TODO (后续RAG阈值调参)
         0.7 or 0.45如何给的?不应该拍着脑袋给。压测得到（构建数据集【询问的方式、llm提取到的商品名】 2. 构建阈值集 3.跑完整个流程：得到哪些阈值更适合构建的数据集）

        confirmed容器不应该出现两个一模一样的商品名--->下游检索【根本不需要两个一模一样的商品名】
        options容器中不能出现两个一模一样的商品名--->用户展示[也不应该展示两个一模一样的商品名]
        场景：confirmed中的商品名可能在options出现：如果某个商品名在confirmed中出现，不用在添加到options中。
        记住：
        像options中添加的商品名满足三个条件 条件1：小于high阈值且大于options 条件2：该商品名不在options 条件3：不在confirmed中
        像confirmed中添加商品名的:条件1：阈值大于high(3个小条件) 条件2：该商品名不在confirmed中 条件3：不用考虑options的，哪怕这个商品名已经在options.只要能进入不到
        confirmed中添加



        Args:
            search_result:  向量数据库检索到的结果

        Returns:
          最终两个容器confirmed、options列表中的商品名
        """

        # 1. 定义两个容器
        confirmed = []
        options = []

        # 2. 遍历检索到的所有商品名的从milvus中的搜索结果
        for item_sea_res in search_result:
            # 2.1 获取extracted_name（LLM提取的商品名）
            llm_extract_item_name = item_sea_res.get('extracted_name')

            # 2.2 获取matches(某一个商品名下的搜索结果)
            item_name_matches = item_sea_res.get('matches')

            # 2.3 (可选) 对某一个商品名下的搜索结果根据分数进行降序排序
            item_name_matches_sorted = sorted(item_name_matches, key=lambda x: x['score'], reverse=True)

            # 2.4 (收集--->分数值大于0.7的搜索结果:高置信度)
            high = [h for h in item_name_matches_sorted if h.get('score') > self._config.item_name_high_confidence]

            # 2.5 高置信度的有
            if high:
                # 1) 是否从向量数据库中搜索到的商品名还和LLM提取到的相等，如果想等，最精准
                extract = next((h for h in high if h.get('item_name') == llm_extract_item_name), None)

                if extract:  # 条件很难触发(有可能 概率低)
                    picked = extract.get('item_name')
                    if picked not in confirmed:  # 去重（认为有重复的 去重）
                        confirmed.append(picked)  # 精确匹配
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)  # 仅有一个高置信度的商品名，直接放到confirmed中
                else:
                    # 有多个分数值比较高的商品名[0.75,0.74,0.71]
                    top_score = high[0]['score']
                    if top_score - high[1]['score'] >= self._config.item_name_score_gap:
                        picked = high[0]['item_name']
                        if picked not in confirmed:
                            confirmed.append(picked)  # 有多个高置信度的商品名，分数差大于0.08，把分数最高的的商品名放到confirmed中
                    else:
                        for h in high[:self._config.item_name_max_options]:  # 一次循环内部只留三个可选的
                            picked = h.get('item_name')
                            if picked not in options and picked not in confirmed:
                                options.append(picked)  # 有多个高置信度的商品名，分数差小于0.08，放到options中
            # 不是高置信度，有可能是中等置信，也有可能不是中等置信
            else:
                mid = [m for m in item_name_matches_sorted if
                       m['score'] >= self._config.item_name_mid_confidence
                       and m.get('item_name') not in options
                       and m.get('item_name') not in confirmed]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get('item_name'))  # 一次循环内部只留三个可选的

        return confirmed, options[:self._config.item_name_max_options]  # 最终只留三个可选的商品名给前端

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):

        """
        # RS12数字万用表和RS13数字万用表
        item_name:RS12数字万用表:0.90
        item_name:万用表:0.71
        item_name:RS13数字万用表:0.88
        Args:
            confirmed:
            search_result:[]

        Returns:

        """

        # 1. 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for search_result in search_result:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed
        if not item_name_score:
            return confirmed

        # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if
                max_score - score <= self._config.item_name_score_gap]

class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm_node"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        主要职责：
        1. 利用LLM从用户原始查询中提取商品名以及改写原始问查询（我喜欢你）
        1.1 如果LLM提取到了商品名，才进行第2步 去milvus对齐
        1.2 如果LLM没有提取到商品名，直接返回
        2. 根据Milvus中存储的商品名进行对齐（目的：检索更加的准确：三路检索都会利用该节点提取到的商品名，因此直接用LLM提取到商品名的话 下游三路检索在过滤的时候，过滤条件极其不准确。导致检索到的噪音很多 LLM最终输出的幻觉很高）
        最终不是要LLM的商品名 而是要Milvus中存储的商品名：因为milvus中没一个chunK都会关联milvus自己的商品名
        3. 决策（该走下去，还是回头）

        利用两个容器，产生三个分支：第一个分支去检索  第二个分支：给用户确认  第三个分支：抱歉
        1. confirmed:如果是精确的商品名--->给confirmed添加精确的商品名
        2. options:商品名不是精确，可是找到多个相似的---->给options中添加找到的多个不精确的商品名。

        state['answer']不要给，进行三路检索
        获取到三路检索结果
        把三路检索到的结果(RRF  RERANKER)给LLM
        LLM生成答案,在state['answer']
        state['answer']:就返回：
        1. 返回候选商品名【不精确】，给用户下一步确认使用
        2. 没有任何商品名，返回抱歉，没有找到您询问的关于任何商品的名字

        Args:
            state: 查询图状态。

        Returns:

        """
        #1. 从状态中获取原始查询和会话ID
        original_query = state.get("original_query")
        session_id = state.get('session_id')

        # 2. 读取历史（后面讲）TODO
        chat_history = ""

        #3. 调用 LLM 提取商品名称
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, chat_history)

        item_names = clean_llm_result.get('item_names')
        rewritten_query = clean_llm_result.get('rewritten_query')

        # 3. 向量对齐（含分数差异过滤）
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 4. 状态决策
        self._decide(state, item_names, confirmed, options, rewritten_query)

        return state

    def _decide(self, state, item_names, confirmed, options, rewritten_query):
        """
        状态决策函数。

        1. 确认成功：设置 item_names 和 rewritten_query。
        2. 有候选：设置 answer 为反问文案。
        3. 无匹配：设置 answer 为无法识别文案。

        Args:
            state: 查询图状态。
            item_names: 原始提取的商品名称。
            confirmed: 确认的商品名称。
            options: 候选商品名称。
            rewritten_query: 重写后的查询。

        Returns:
        """
        if confirmed:
            #  确认成功：设置 item_names 和 rewritten_query
            # state 中没有 answer → main_graph 条件边走 "继续检索"
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query

        elif options:
            #  有候选：设置 answer 为反问文案
            # state 中有 answer → main_graph 条件边走 "直接返回"
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )

        else:
            # 无匹配：设置 answer 为无法识别文案
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == "__main__":
    node = ItemNameConfirmNode()
    init_state={
        "original_query": "RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛"
    }
    result = node.process(init_state)
    print(result)
