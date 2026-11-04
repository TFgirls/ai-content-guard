"""Agent 循环测试。

重点是**多轮路径**：真实评测集里 18 张全部第 1 轮就收敛，
「第 2 轮」那段代码从来没被真实数据走过，只能靠这里保证它是对的。
"""

from src.agent import review_text
from src.rewrite import RewriteOutcome


class Test终止条件:
    def test_一轮改干净(self, scripted_rewrite):
        scripted_rewrite(["三天缓解颈椎不适"])
        t = review_text("三天根治颈椎病，无效退款")
        assert t.resolved is True
        assert t.attempts == 1
        assert len(t.rounds) == 1
        assert t.rounds[0].residual == []

    def test_第二轮才改干净(self, scripted_rewrite):
        """第 1 轮故意留下违规词，验证循环会带着残留再改一次。"""
        _, state = scripted_rewrite(["三天根治颈椎病，还是违规", "三天缓解颈椎不适"])
        t = review_text("三天根治颈椎病，无效退款")

        assert state["calls"] == 2
        assert t.resolved is True
        assert t.attempts == 2
        assert len(t.rounds) == 2
        assert len(t.rounds[0].residual) > 0      # 第 1 轮有残留
        assert t.rounds[1].residual == []          # 第 2 轮清零
        assert "第 2 轮" in t.summary()

    def test_跑满上限转人工(self, scripted_rewrite):
        scripted_rewrite(["三天根治颈椎病"] * 5)
        t = review_text("三天根治颈椎病", max_rounds=3)

        assert t.resolved is False
        assert t.escalated is True
        assert t.attempts == 3
        assert "最大改写轮次" in t.reason

    def test_改写失败立刻停不空转(self, scripted_rewrite):
        """没 Key / 网络失败时重试没有意义，应一轮就转人工。"""
        fail = RewriteOutcome("原文", ["失败"], ok=False, error="missing_api_key")
        _, state = scripted_rewrite([fail, fail, fail])

        t = review_text("三天根治颈椎病")
        assert state["calls"] == 1
        assert t.escalated is True
        assert "missing_api_key" in t.reason

    def test_上限设为1时一轮不干净就转人工(self, scripted_rewrite):
        scripted_rewrite(["三天根治颈椎病"] * 3)
        t = review_text("三天根治颈椎病", max_rounds=1)
        assert t.escalated is True
        assert t.attempts == 1

    def test_空文本不进入循环(self, scripted_rewrite):
        _, state = scripted_rewrite(["x"])
        t = review_text("   ")
        assert state["calls"] == 0
        assert t.escalated is True


class Test规则漏检场景:
    def test_规则零命中但判定违规时强制改写(self, scripted_rewrite):
        """hard_01 类样本：模型说违规、词表一条没命中。

        循环如果只看规则扫描，会直接判「无需改写」把违规内容放过去。
        """
        _, state = scripted_rewrite(["三天感受变化，日常调理也能帮上忙"])
        t = review_text("三天见效，不吃药也能好", ["医疗功效"], initial_violation=True)

        assert state["calls"] == 1
        assert t.resolved is True
        assert t.rounds[0].hits == []          # 规则层确实没命中
        assert t.final_text != "三天见效，不吃药也能好"

    def test_无违规时不动用改写(self, scripted_rewrite):
        _, state = scripted_rewrite(["不该被调用"])
        t = review_text("秋季新款，舒适透气", initial_violation=False)

        assert state["calls"] == 0
        assert t.attempts == 0
        assert t.resolved is True


class Test观测信息:
    def test_首轮命中数可读(self, scripted_rewrite):
        scripted_rewrite(["干净文案"])
        t = review_text("三天根治颈椎病，无效退款")
        assert t.first_hit_count == 2

    def test_每轮记录命中与残留(self, scripted_rewrite):
        scripted_rewrite(["三天根治颈椎病", "三天缓解颈椎不适"])
        t = review_text("三天根治颈椎病，无效退款")

        assert t.rounds[0].round == 1
        assert len(t.rounds[0].hits) == 2
        assert len(t.rounds[0].residual) == 1
        assert t.rounds[1].round == 2
        assert len(t.rounds[1].residual) == 0

    def test_summary可直接展示(self, scripted_rewrite):
        scripted_rewrite(["干净"])
        assert review_text("三天根治颈椎病").summary().startswith("第 1 轮")


class Test时间预算:
    def test_预算耗尽时不再开新一轮(self, scripted_rewrite):
        """第 1 轮改不干净，但剩余预算不足半次调用，应直接转人工。"""
        _, state = scripted_rewrite(["三天根治颈椎病"] * 5)
        t = review_text("三天根治颈椎病，无效退款", budget=0.001)

        assert state["calls"] == 1          # 只跑了第 1 轮
        assert t.escalated is True
        assert "预算" in t.reason

    def test_预算充足时正常多轮(self, scripted_rewrite):
        _, state = scripted_rewrite(["三天根治颈椎病", "三天缓解颈椎不适"])
        t = review_text("三天根治颈椎病", budget=60)

        assert state["calls"] == 2
        assert t.resolved is True


class Test边界_最后一轮才收敛:
    """回归用例：曾漏测这个中间态，导致「仍残留 0 处」的自相矛盾结论。"""

    def test_最后一轮改干净应判为通过(self, scripted_rewrite):
        """limit=3，前两轮有残留、第三轮清零——这是收敛，不是转人工。"""
        _, state = scripted_rewrite(["三天根治颈椎病", "三天根治颈椎病", "三天缓解颈椎不适"])
        t = review_text("三天根治颈椎病，无效退款", max_rounds=3)

        assert state["calls"] == 3
        assert t.resolved is True
        assert t.escalated is False
        assert t.rounds[-1].residual == []
        assert "残留 0 处" not in t.summary()

    def test_跑满且真有残留才转人工(self, scripted_rewrite):
        """对照组：三轮都改不干净，仍要转人工。"""
        scripted_rewrite(["三天根治颈椎病"] * 5)
        t = review_text("三天根治颈椎病", max_rounds=3)

        assert t.resolved is False
        assert t.escalated is True
        assert len(t.rounds[-1].residual) > 0
