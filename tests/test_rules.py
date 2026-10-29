"""规则层测试：归一化、大小写、数字边界、风险分级。

这些用例基本都是从真实踩过的坑反推出来的，改动 rules.py 时跑一遍
就能知道有没有把老问题带回来。
"""

from src.rules import normalize, rule_risk, scan_rules


class TestNormalize:
    def test_全角转半角(self):
        assert normalize("１００％")[0] == "100%"

    def test_去空白与换行(self):
        assert normalize("无效 退\n款")[0] == "无效退款"

    def test_去零宽字符(self):
        assert normalize("无\u200b效")[0] == "无效"

    def test_索引能映射回原文(self):
        text = "无 效 退 款"
        norm, idx = normalize(text)
        assert norm == "无效退款"
        # 第 0 个归一化字符来自原文下标 0，第 3 个来自下标 6
        assert text[idx[0]] == "无"
        assert text[idx[3]] == "款"

    def test_空文本(self):
        assert normalize("") == ("", [])


class TestScanRules:
    def test_跨行断词仍能命中(self):
        """OCR 把词拆到两行，不做归一化就会漏。"""
        hits = scan_rules("三天根治颈椎病，无效退\n款")
        assert any(h["match"] == "无效退款" for h in hits)

    def test_关闭归一化会漏掉跨行断词(self):
        hits = scan_rules("无效退\n款", normalize_text=False)
        assert not any(h["match"] == "无效退款" for h in hits)

    def test_字间空格仍能命中(self):
        hits = scan_rules("无效 退款，放心下单")
        assert any(h["match"] == "无效退款" for h in hits)

    def test_全角数字仍能命中(self):
        hits = scan_rules("１００％有效")
        assert any(h["match"] == "100%" for h in hits)

    def test_词表大小写不敏感(self):
        """词表只收「加V」，小写写法也要能对上。"""
        hits = scan_rules("加v：wxid8899")
        assert any(h["category"] == "违规导流" for h in hits)

    def test_正则大小写不敏感(self):
        assert scan_rules("Vx：abc1234")
        assert scan_rules("vx：abc1234")

    def test_手机号需要数字边界(self):
        """长订单号里切出来的 11 位数字不该当手机号。"""
        assert not scan_rules("订单号 1380013800012345 已发货")

    def test_独立手机号能命中(self):
        hits = scan_rules("联系 13800138000")
        assert any("13800138000" in h["match"] for h in hits)

    def test_命中项带回原文片段(self):
        hits = scan_rules("三天根治颈椎病")
        hit = next(h for h in hits if h["match"] == "根治")
        assert hit["evidence"] == "根治"

    def test_空文本不报错(self):
        assert scan_rules("") == []
        assert scan_rules("   \n  ") == []

    def test_合规文案零命中(self):
        assert scan_rules("秋季新款，舒适透气，支持七天无理由退换") == []


class TestRuleRisk:
    def test_无命中为低风险(self):
        assert rule_risk([]) == "low"

    def test_极限词为中风险(self):
        hits = scan_rules("全网最低价")
        assert rule_risk(hits) == "medium"

    def test_医疗功效为高风险(self):
        hits = scan_rules("根治颈椎病")
        assert rule_risk(hits) == "high"
