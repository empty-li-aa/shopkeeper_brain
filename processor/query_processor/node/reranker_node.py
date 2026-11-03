from typing import Tuple, List, Dict, Any
import math
from FlagEmbedding import FlagReranker
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    name = "reranker_node"

    @staticmethod
    def _sigmoid(score: float) -> float:
        """sigmoid归一化，将 (-∞, +∞) 映射到 (0, 1)"""
        return 1.0 / (1.0 + math.exp(-score))

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """

        Args:
            state:

        Returns:

        """

        # 1. 获取用户问题
        user_query = state.get('rewritten_query') or state.get('original_query')

        # 2. 获取两路检索结果(本地检索结果、远程检索结果)
        rerank_outputs: List[Dict[str, Any]] = self._collect_rerank_inputs(state)

        # 3. 利用Reranker进行精排
        refine_docs: List[Dict[str, Any]] = self._refine_rank(user_query, rerank_outputs)

        # 4.top_k: 静态top_k 设置过大或者过小都有问题。能否使用动态top_k
        # 动态top_k:不是初始的时候不给top_k,初始的时候该给还是给
        # min_top_k(3):不管如何裁减，至上都要留下三个
        # max_top_k(10):最多留下10个
        # 不过这并不是最终的（可能会调整）【尽量减少幻觉，完全解决幻觉】
        #  top_k:如果设置大了 1.LLM上下文过长    2. 可能截取无关上下文(不该检索的确检索到了)--->LLM---> (幻觉)
        #  top_k:如果设置小了 1.LLM上下不会过程  2. 检索到上下文信息不全面(该检索的没有检索到)--->LLM--->(幻觉)

        reranked_docs = self._cliff_cutoff(refine_docs, self.config.rerank_min_top_k, self.config.rerank_max_top_k)

        state['reranked_docs'] = reranked_docs

        return state

    def _collect_rerank_inputs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """
        获取两路检索结果(本地检索结果、远程检索结果)
        Args:
            state:

        Returns:

        """
        final_docs = []
        # 1. 获取本地检索结果
        rrf_chunks = state.get('rrf_chunks') or []
        for chunk in rrf_chunks:
            # 1. 判断chunk
            if not chunk or not isinstance(chunk, dict):
                continue

            # 2. 获取chunk的信息
            # 2.1 获取chunk中content
            content = chunk.get('content', '')
            if not content:
                continue
            # 2.2 获取chunk中的title
            title = chunk.get('title', '')

            # 2.3 获取chunk中的chunk_id(一定有)
            chunk_id = chunk.get('chunk_id')

            # 3. 格式化文档(格式化本地)
            formated_local_doc = self._format_doc(content=content, chunk_id=chunk_id, title=title, source="local")

            final_docs.append(formated_local_doc)

        # 2. 获取远程检索结果
        web_search_docs = state.get('web_search_docs') or []
        for doc in web_search_docs:

            # 1. 判断doc
            if not doc or not isinstance(doc, dict):
                continue

            # 2. 获取content
            content = doc.get('snippet', '')
            # 2.3 获取title
            title = doc.get('title', '')
            # 2.4 获取url
            url = doc.get('url', '')

            # 3. 格式化文档(格式化远程)
            formated_web_doc = self._format_doc(content=content, title=title, url=url, source="web")
            final_docs.append(formated_web_doc)

        self.logger.info(f"获取Reranker阶段需要的搜索结果个数{len(final_docs)}")
        return final_docs

    def _format_doc(self, content: str, chunk_id: int = None, title: str = "", url: str = "", source: str = ""):
        """
        格式化本地以及远程检索到的文档
        Args:
            content:
            chunk_id:
            title:
            source:
            url:

        Returns:

        """
        return {
            "content": content,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "source": source
        }

    def _refine_rank(self, user_query: str, rerank_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        reranker模型进行打分&排序【精排】
        Args:
            user_query:  用户的查询
            rerank_outputs: 本地和远程融合后的检索结果

        Returns:
            Dict[str,Any]:{"score","","other":"..."}

        """
        if not rerank_outputs:
            return []
        # 1. 获取重排序模型
        try:
            rerank_client: FlagReranker = AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.error(f"获取BGE-M3重排序模型失败 原因:{str(e)}")
            return [{**d, "score": None} for d in rerank_outputs]

        # 2. 构建Q->D的pair对
        query_doc_pairs = [(user_query, d.get('content')) for d in rerank_outputs]

        try:
            # 3. 计算(注意：BGE-M3重排序模型计算出来的得分可以很大也可以很小(负无穷大,正无穷大))
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_pairs)

            # 4.组合最终结果
            doc_score = [{**d, "score": self._sigmoid(float(s))} for d, s in zip(rerank_outputs, rerank_scores)]

            # 5. 排序
            sorted_doc_score = sorted(doc_score, key=lambda x: x['score'], reverse=True)

            # 6. 返回
            return sorted_doc_score
        except Exception as e:
            self.logger.error(f"BGE-M3重排序模型计算分数失败 原因：{str(e)}")
            return [{**d, "score": None} for d in rerank_outputs]

    def _cliff_cutoff(self, refine_docs: List[Dict[str, Any]], rerank_min_top_k: int, rerank_max_top_k: int) -> List[
        Dict[str, Any]]:
        """
          动态top_k: 归一化后只需一个断崖阈值(rerank_gap_threshold)
          从头开始寻找最大断崖点，再用 min_top_k 兜底
        """

        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        cut_off = upper_bound

        # 1. 相关性硬下限：sigmoid 后分数就是 P(相关)，0.5 = "更可能相关"的分界；
        #    低于 0.5 的先砍掉（概率有物理含义，不依赖分数绝对值，不会再出现"太大或负数"）
        for i in range(upper_bound):
            score = refine_docs[i].get('score')
            if score is not None and score < 0.5:
                cut_off = i
                break

        # 兜底：不管断崖在哪，至少保留 lower_bound 个
        cut_off = max(cut_off, lower_bound)

        # 从第0个间隔开始遍历，找全局最大断崖 —— 改用"拐点"实现：
        # 看整条分数曲线的几何形状（由陡变平的弯角），参数无关，分数整体平移/缩放位置不变
        knee_index = self._knee_index([d.get('score') or 0.0 for d in refine_docs[:cut_off]])
        if knee_index < cut_off:
            self.logger.info(f"位置{knee_index}发生断崖")
            cut_off = knee_index

        # 兜底：拐点检测后，再次保证至少保留 lower_bound 个
        cut_off = max(cut_off, lower_bound)

        cutoff_docs = refine_docs[:cut_off]

        return cutoff_docs

    @staticmethod
    def _knee_index(scores: List[float]) -> int:
        """
        递减曲线 scores 的"拐点"下标：
        取到首尾连线垂直距离最大的点，返回拐点后一位作为切点。
        （参数无关：只依赖曲线的相对形状，不依赖分数绝对大小）
        """
        n = len(scores)
        if n < 3:
            return n  # 点太少，谈不上拐点

        x0, y0 = 0, scores[0]
        x1, y1 = n - 1, scores[-1]
        denom = math.hypot(x1 - x0, y1 - y0)
        if denom == 0:
            return n  # 所有分数相等，没有断崖

        span = abs(y1 - y0)  # 分数总跨度

        best_i, best_d = n, 0.0
        for i in range(1, n - 1):
            # 点 (i, scores[i]) 到首尾线段 (x0,y0)-(x1,y1) 的垂直距离
            d = abs((x1 - x0) * (y0 - scores[i]) - (x0 - i) * (y1 - y0)) / denom
            if d > best_d:
                best_d, best_i = d, i

        # 显著性：拐点的凸起程度必须达到分数跨度的 10% 才算真断崖；
        # 否则视为“接近直线 / 浮点噪声”（例如全相关、分数均匀缓降），返回 n（不裁）。
        # 这个阈值是“相对跨度”的，尺度无关，不随分数绝对大小变化。
        if span <= 0 or best_d < 0.1 * span:
            return n

        return best_i + 1




if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)
    print(result)
