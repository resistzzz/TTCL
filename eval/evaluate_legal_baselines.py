
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

"""Versioned deterministic legal EM. Shared by training and evaluation."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from collections.abc import Sequence

v1_VERSION='legal_shortqa_em_v1'
@lru_cache(maxsize=8)
def v1_vocabulary(sha):
    if sha != VOCAB_SHA256:
        raise ValueError('Unknown frozen vocabulary hash')
    return json.loads(VOCAB_JSON)
def v1_surface(s):
    s=unicodedata.normalize('NFKC',s).casefold().strip()
    return re.sub(r'\s+',' ',s).replace('‑','-').replace('–','-').strip(' .。!！\"\'“”‘’')
def v1_number(s):
    if s.isdigit():return int(s)
    values={c:i for i,c in enumerate('零一二三四五六七八九')};total=current=0
    for c in s:
        if c in values:current=values[c]
        elif c in '十百千':total+=(current or 1)*{'十':10,'百':100,'千':1000}[c];current=0
        else:raise ValueError(s)
    return total+current
v1_N=r'[0-9零一二三四五六七八九十百千]+'
v1_ARTICLE=re.compile(r'(?:(中华人民共和国刑法|中华人民共和国民法典|刑法|民法典))?第('+v1_N+r'(?:、'+v1_N+r')*)条(?:第('+v1_N+r')款)?(?:第('+v1_N+r')项)?')
def v1_articles(s):
    s=v1_surface(s).replace('《','').replace('》','');law=None;pos=0;result=[]
    for m in v1_ARTICLE.finditer(s):
        if re.sub(r'[\s、,;和及与]','',s[pos:m.start()]):return None
        if m[1]:law='刑法' if '刑法' in m[1] else '民法典'
        if not law:return None
        for n in m[2].split('、'):result.append((law,v1_number(n),v1_number(m[3]) if m[3] else None,v1_number(m[4]) if m[4] else None))
        pos=m.end()
    if s[pos:] not in ('','的规定','规定'):return None
    return tuple(sorted(set(result),key=str)) if result else None
def v1_crime_parts(s):return [p.strip() for p in re.split(r';|(?<=罪)[、,和]',s) if p.strip()]
def v1_labels(s,v1_vocabulary):
    s=v1_surface(s);keys=sorted(v1_vocabulary,key=len,reverse=True);pos=0;result=[]
    while pos<len(s):
        found=False
        for key in keys:
            if not s.startswith(key,pos):continue
            end=pos+len(key)
            if end==len(s):result.append(v1_vocabulary[key]);pos=end;found=True;break
            sep=re.match(r'(?:\s*[,;、]\s*|\s+(?:and|&)\s+|[和及与])',s[end:])
            if sep:result.append(v1_vocabulary[key]);pos=end+sep.end();found=True;break
        if not found:return None
    return tuple(sorted(set(result))) if result else None
def v1_canonical(s,task,vocab):
    if not isinstance(s,str) or not s.strip():return None
    if task=='statute_article_grounding':return v1_articles(s)
    if task=='case_outcome_judgment':
        m=re.fullmatch(r'(?:(?:the )?appeal (?:(?:is|was|has been) )?)?(allowed|dismissed)',v1_surface(s))
        return m[1] if m else None
    if task in vocab:return v1_labels(s,vocab[task])
    return v1_surface(s)
def v1_compute_score(solution_str,ground_truth,extra_info=None):
    info=extra_info or {}
    if info.get('legal_em_version')!=v1_VERSION:raise ValueError('Legal EM version missing or mismatched')
    task=info['legal_task'];vocab={t:dict(v) for t,v in v1_vocabulary(info['legal_vocab_sha256']).items()}
    refs=list(ground_truth) if isinstance(ground_truth,Sequence) and not isinstance(ground_truth,(str,bytes)) else [ground_truth]
    # Unknown held-out label spellings can match their own gold, but do not add
    # aliases from another eval record or extract labels from long explanations.
    for ref in refs:
        if task in vocab and isinstance(ref,str):
            for atom in v1_crime_parts(ref) if task=='criminal_charge_judgment' else ref.split(';'):
                key=v1_surface(atom);vocab[task].setdefault(key,key)
                if task=='criminal_charge_judgment':
                    base=key[:-1] if key.endswith('罪') else key
                    vocab[task][base]=base;vocab[task][base+'罪']=base
    if not isinstance(solution_str,str):return 0.
    prediction=v1_canonical(solution_str,task,vocab)
    for ref in refs:
        if not isinstance(ref,str):continue
        if solution_str.strip()==ref.strip():return 1.
        target=v1_canonical(ref,task,vocab)
        if prediction is not None and target is not None and prediction==target:return 1.
    return 0.

VOCAB_SHA256 = '1779077ab385c8157fc93a5439759945d9d701e1538d533a33362d2ca4e114ae'
VOCAB_JSON = '{\n  "contract_type_judgment": {\n    "保管合同": "保管合同",\n    "房屋租赁合同": "房屋租赁合同",\n    "租赁合同": "租赁合同"\n  },\n  "criminal_charge_judgment": {\n    "串通投标": "串通投标",\n    "串通投标罪": "串通投标",\n    "为境外非法提供国家秘密": "为境外非法提供国家秘密",\n    "为境外非法提供国家秘密罪": "为境外非法提供国家秘密",\n    "介绍贿赂": "介绍贿赂",\n    "介绍贿赂罪": "介绍贿赂",\n    "传授犯罪方法": "传授犯罪方法",\n    "传授犯罪方法罪": "传授犯罪方法",\n    "传播性病": "传播性病",\n    "传播性病罪": "传播性病",\n    "传播淫秽物品": "传播淫秽物品",\n    "传播淫秽物品罪": "传播淫秽物品",\n    "伪证": "伪证",\n    "伪证罪": "伪证",\n    "伪造、倒卖伪造的有价票证": "伪造、倒卖伪造的有价票证",\n    "伪造、倒卖伪造的有价票证罪": "伪造、倒卖伪造的有价票证",\n    "伪造、变造、买卖国家机关公文、证件、印章": "伪造、变造、买卖国家机关公文、证件、印章",\n    "伪造、变造、买卖国家机关公文、证件、印章罪": "伪造、变造、买卖国家机关公文、证件、印章",\n    "伪造、变造、买卖武装部队公文、证件、印章": "伪造、变造、买卖武装部队公文、证件、印章",\n    "伪造、变造、买卖武装部队公文、证件、印章罪": "伪造、变造、买卖武装部队公文、证件、印章",\n    "伪造、变造居民身份证": "伪造、变造居民身份证",\n    "伪造、变造居民身份证罪": "伪造、变造居民身份证",\n    "伪造公司、企业、事业单位、人民团体印章": "伪造公司、企业、事业单位、人民团体印章",\n    "伪造公司、企业、事业单位、人民团体印章罪": "伪造公司、企业、事业单位、人民团体印章",\n    "伪造货币": "伪造货币",\n    "伪造货币罪": "伪造货币",\n    "侮辱": "侮辱",\n    "侮辱罪": "侮辱",\n    "侵占": "侵占",\n    "侵占罪": "侵占",\n    "侵犯著作权": "侵犯著作权",\n    "侵犯著作权罪": "侵犯著作权",\n    "保险诈骗": "保险诈骗",\n    "保险诈骗罪": "保险诈骗",\n    "信用卡诈骗": "信用卡诈骗",\n    "信用卡诈骗罪": "信用卡诈骗",\n    "倒卖文物": "倒卖文物",\n    "倒卖文物罪": "倒卖文物",\n    "倒卖车票、船票": "倒卖车票、船票",\n    "倒卖车票、船票罪": "倒卖车票、船票",\n    "假冒注册商标": "假冒注册商标",\n    "假冒注册商标罪": "假冒注册商标",\n    "冒充军人招摇撞骗": "冒充军人招摇撞骗",\n    "冒充军人招摇撞骗罪": "冒充军人招摇撞骗",\n    "出售、购买、运输假币": "出售、购买、运输假币",\n    "出售、购买、运输假币罪": "出售、购买、运输假币",\n    "利用影响力受贿": "利用影响力受贿",\n    "利用影响力受贿罪": "利用影响力受贿",\n    "制造、贩卖、传播淫秽物品": "制造、贩卖、传播淫秽物品",\n    "制造、贩卖、传播淫秽物品罪": "制造、贩卖、传播淫秽物品",\n    "动植物检疫徇私舞弊": "动植物检疫徇私舞弊",\n    "动植物检疫徇私舞弊罪": "动植物检疫徇私舞弊",\n    "劫持船只、汽车": "劫持船只、汽车",\n    "劫持船只、汽车罪": "劫持船只、汽车",\n    "包庇毒品犯罪分子": "包庇毒品犯罪分子",\n    "包庇毒品犯罪分子罪": "包庇毒品犯罪分子",\n    "协助组织卖淫": "协助组织卖淫",\n    "协助组织卖淫罪": "协助组织卖淫",\n    "单位受贿": "单位受贿",\n    "单位受贿罪": "单位受贿",\n    "单位行贿": "单位行贿",\n    "单位行贿罪": "单位行贿",\n    "危险物品肇事": "危险物品肇事",\n    "危险物品肇事罪": "危险物品肇事",\n    "危险驾驶": "危险驾驶",\n    "危险驾驶罪": "危险驾驶",\n    "受贿": "受贿",\n    "受贿罪": "受贿",\n    "合同诈骗": "合同诈骗",\n    "合同诈骗罪": "合同诈骗",\n    "国家机关工作人员签订、履行合同失职被骗": "国家机关工作人员签订、履行合同失职被骗",\n    "国家机关工作人员签订、履行合同失职被骗罪": "国家机关工作人员签订、履行合同失职被骗",\n    "失火": "失火",\n    "失火罪": "失火",\n    "妨害作证": "妨害作证",\n    "妨害作证罪": "妨害作证",\n    "妨害信用卡管理": "妨害信用卡管理",\n    "妨害信用卡管理罪": "妨害信用卡管理",\n    "妨害公务": "妨害公务",\n    "妨害公务罪": "妨害公务",\n    "容留他人吸毒": "容留他人吸毒",\n    "容留他人吸毒罪": "容留他人吸毒",\n    "对单位行贿": "对单位行贿",\n    "对单位行贿罪": "对单位行贿",\n    "对非国家工作人员行贿": "对非国家工作人员行贿",\n    "对非国家工作人员行贿罪": "对非国家工作人员行贿",\n    "寻衅滋事": "寻衅滋事",\n    "寻衅滋事罪": "寻衅滋事",\n    "巨额财产来源不明": "巨额财产来源不明",\n    "巨额财产来源不明罪": "巨额财产来源不明",\n    "帮助犯罪分子逃避处罚": "帮助犯罪分子逃避处罚",\n    "帮助犯罪分子逃避处罚罪": "帮助犯罪分子逃避处罚",\n    "开设赌场": "开设赌场",\n    "开设赌场罪": "开设赌场",\n    "引诱、容留、介绍卖淫": "引诱、容留、介绍卖淫",\n    "引诱、容留、介绍卖淫罪": "引诱、容留、介绍卖淫",\n    "引诱、教唆、欺骗他人吸毒": "引诱、教唆、欺骗他人吸毒",\n    "引诱、教唆、欺骗他人吸毒罪": "引诱、教唆、欺骗他人吸毒",\n    "强制猥亵、侮辱妇女": "强制猥亵、侮辱妇女",\n    "强制猥亵、侮辱妇女罪": "强制猥亵、侮辱妇女",\n    "强奸": "强奸",\n    "强奸罪": "强奸",\n    "强迫交易": "强迫交易",\n    "强迫交易罪": "强迫交易",\n    "强迫他人吸毒": "强迫他人吸毒",\n    "强迫他人吸毒罪": "强迫他人吸毒",\n    "强迫劳动": "强迫劳动",\n    "强迫劳动罪": "强迫劳动",\n    "徇私枉法": "徇私枉法",\n    "徇私枉法罪": "徇私枉法",\n    "打击报复证人": "打击报复证人",\n    "打击报复证人罪": "打击报复证人",\n    "投放危险物质": "投放危险物质",\n    "投放危险物质罪": "投放危险物质",\n    "抢劫": "抢劫",\n    "抢劫罪": "抢劫",\n    "抢夺": "抢夺",\n    "抢夺罪": "抢夺",\n    "拐卖妇女、儿童": "拐卖妇女、儿童",\n    "拐卖妇女、儿童罪": "拐卖妇女、儿童",\n    "拐骗儿童": "拐骗儿童",\n    "拐骗儿童罪": "拐骗儿童",\n    "拒不执行判决、裁定": "拒不执行判决、裁定",\n    "拒不执行判决、裁定罪": "拒不执行判决、裁定",\n    "拒不支付劳动报酬": "拒不支付劳动报酬",\n    "拒不支付劳动报酬罪": "拒不支付劳动报酬",\n    "招摇撞骗": "招摇撞骗",\n    "招摇撞骗罪": "招摇撞骗",\n    "持有、使用假币": "持有、使用假币",\n    "持有、使用假币罪": "持有、使用假币",\n    "挪用公款": "挪用公款",\n    "挪用公款罪": "挪用公款",\n    "挪用特定款物": "挪用特定款物",\n    "挪用特定款物罪": "挪用特定款物",\n    "挪用资金": "挪用资金",\n    "挪用资金罪": "挪用资金",\n    "掩饰、隐瞒犯罪所得、犯罪所得收益": "掩饰、隐瞒犯罪所得、犯罪所得收益",\n    "掩饰、隐瞒犯罪所得、犯罪所得收益罪": "掩饰、隐瞒犯罪所得、犯罪所得收益",\n    "提供侵入、非法控制计算机信息系统程序、工具": "提供侵入、非法控制计算机信息系统程序、工具",\n    "提供侵入、非法控制计算机信息系统程序、工具罪": "提供侵入、非法控制计算机信息系统程序、工具",\n    "收买被拐卖的妇女、儿童": "收买被拐卖的妇女、儿童",\n    "收买被拐卖的妇女、儿童罪": "收买被拐卖的妇女、儿童",\n    "放火": "放火",\n    "放火罪": "放火",\n    "故意伤害": "故意伤害",\n    "故意伤害罪": "故意伤害",\n    "故意杀人": "故意杀人",\n    "故意杀人罪": "故意杀人",\n    "故意毁坏财物": "故意毁坏财物",\n    "故意毁坏财物罪": "故意毁坏财物",\n    "敲诈勒索": "敲诈勒索",\n    "敲诈勒索罪": "敲诈勒索",\n    "构成受贿": "构成受贿",\n    "构成受贿罪": "构成受贿",\n    "构成非法拘禁": "构成非法拘禁",\n    "构成非法拘禁罪": "构成非法拘禁",\n    "污染环境": "污染环境",\n    "污染环境罪": "污染环境",\n    "滥伐林木": "滥伐林木",\n    "滥伐林木罪": "滥伐林木",\n    "滥用职权": "滥用职权",\n    "滥用职权罪": "滥用职权",\n    "滥用职权罪或玩忽职守": "滥用职权罪或玩忽职守",\n    "滥用职权罪或玩忽职守罪": "滥用职权罪或玩忽职守",\n    "爆炸": "爆炸",\n    "爆炸罪": "爆炸",\n    "猥亵儿童": "猥亵儿童",\n    "猥亵儿童罪": "猥亵儿童",\n    "玩忽职守": "玩忽职守",\n    "玩忽职守罪": "玩忽职守",\n    "生产、销售不符合安全标准的食品": "生产、销售不符合安全标准的食品",\n    "生产、销售不符合安全标准的食品罪": "生产、销售不符合安全标准的食品",\n    "生产、销售伪劣产品": "生产、销售伪劣产品",\n    "生产、销售伪劣产品罪": "生产、销售伪劣产品",\n    "生产、销售假药": "生产、销售假药",\n    "生产、销售假药罪": "生产、销售假药",\n    "生产、销售有毒、有害食品": "生产、销售有毒、有害食品",\n    "生产、销售有毒、有害食品罪": "生产、销售有毒、有害食品",\n    "甲第一个行为构成侵占罪,第二个行为构成诈骗": "甲第一个行为构成侵占罪,第二个行为构成诈骗",\n    "甲第一个行为构成侵占罪,第二个行为构成诈骗罪": "甲第一个行为构成侵占罪,第二个行为构成诈骗",\n    "盗伐林木": "盗伐林木",\n    "盗伐林木罪": "盗伐林木",\n    "盗掘古文化遗址、古墓葬": "盗掘古文化遗址、古墓葬",\n    "盗掘古文化遗址、古墓葬罪": "盗掘古文化遗址、古墓葬",\n    "盗窃": "盗窃",\n    "盗窃、侮辱尸体": "盗窃、侮辱尸体",\n    "盗窃、侮辱尸体罪": "盗窃、侮辱尸体",\n    "盗窃罪": "盗窃",\n    "破坏交通工具": "破坏交通工具",\n    "破坏交通工具罪": "破坏交通工具",\n    "破坏交通设施": "破坏交通设施",\n    "破坏交通设施罪": "破坏交通设施",\n    "破坏广播电视设施、公用电信设施": "破坏广播电视设施、公用电信设施",\n    "破坏广播电视设施、公用电信设施罪": "破坏广播电视设施、公用电信设施",\n    "破坏易燃易爆设备": "破坏易燃易爆设备",\n    "破坏易燃易爆设备罪": "破坏易燃易爆设备",\n    "破坏生产经营": "破坏生产经营",\n    "破坏生产经营罪": "破坏生产经营",\n    "破坏电力设备": "破坏电力设备",\n    "破坏电力设备罪": "破坏电力设备",\n    "破坏监管秩序": "破坏监管秩序",\n    "破坏监管秩序罪": "破坏监管秩序",\n    "破坏计算机信息系统": "破坏计算机信息系统",\n    "破坏计算机信息系统罪": "破坏计算机信息系统",\n    "票据诈骗": "票据诈骗",\n    "票据诈骗罪": "票据诈骗",\n    "私分国有资产": "私分国有资产",\n    "私分国有资产罪": "私分国有资产",\n    "窃取、收买、非法提供信用卡信息": "窃取、收买、非法提供信用卡信息",\n    "窃取、收买、非法提供信用卡信息罪": "窃取、收买、非法提供信用卡信息",\n    "窝藏、包庇": "窝藏、包庇",\n    "窝藏、包庇罪": "窝藏、包庇",\n    "窝藏、转移、隐瞒毒品、毒赃": "窝藏、转移、隐瞒毒品、毒赃",\n    "窝藏、转移、隐瞒毒品、毒赃罪": "窝藏、转移、隐瞒毒品、毒赃",\n    "组织、强迫、引诱、容留、介绍卖淫": "组织、强迫、引诱、容留、介绍卖淫",\n    "组织、强迫、引诱、容留、介绍卖淫罪": "组织、强迫、引诱、容留、介绍卖淫",\n    "组织、领导传销活动": "组织、领导传销活动",\n    "组织、领导传销活动罪": "组织、领导传销活动",\n    "组织卖淫": "组织卖淫",\n    "组织卖淫罪": "组织卖淫",\n    "绑架": "绑架",\n    "绑架罪": "绑架",\n    "编造、故意传播虚假恐怖信息": "编造、故意传播虚假恐怖信息",\n    "编造、故意传播虚假恐怖信息罪": "编造、故意传播虚假恐怖信息",\n    "职务侵占": "职务侵占",\n    "职务侵占罪": "职务侵占",\n    "聚众冲击国家机关": "聚众冲击国家机关",\n    "聚众冲击国家机关罪": "聚众冲击国家机关",\n    "聚众扰乱公共场所秩序、交通秩序": "聚众扰乱公共场所秩序、交通秩序",\n    "聚众扰乱公共场所秩序、交通秩序罪": "聚众扰乱公共场所秩序、交通秩序",\n    "聚众扰乱社会秩序": "聚众扰乱社会秩序",\n    "聚众扰乱社会秩序罪": "聚众扰乱社会秩序",\n    "聚众斗殴": "聚众斗殴",\n    "聚众斗殴罪": "聚众斗殴",\n    "虐待": "虐待",\n    "虐待罪": "虐待",\n    "虐待被监护、看护人": "虐待被监护、看护人",\n    "虐待被监护、看护人罪": "虐待被监护、看护人",\n    "虐待被监管人": "虐待被监管人",\n    "虐待被监管人罪": "虐待被监管人",\n    "虚开发票": "虚开发票",\n    "虚开发票罪": "虚开发票",\n    "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票": "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票",\n    "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票罪": "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票",\n    "虚报注册资本": "虚报注册资本",\n    "虚报注册资本罪": "虚报注册资本",\n    "行贿": "行贿",\n    "行贿罪": "行贿",\n    "诈骗": "诈骗",\n    "诈骗罪": "诈骗",\n    "诬告陷害": "诬告陷害",\n    "诬告陷害罪": "诬告陷害",\n    "贪污": "贪污",\n    "贪污罪": "贪污",\n    "赌博": "赌博",\n    "赌博罪": "赌博",\n    "走私": "走私",\n    "走私、贩卖、运输、制造毒品": "走私、贩卖、运输、制造毒品",\n    "走私、贩卖、运输、制造毒品罪": "走私、贩卖、运输、制造毒品",\n    "走私国家禁止进出口的货物、物品": "走私国家禁止进出口的货物、物品",\n    "走私国家禁止进出口的货物、物品罪": "走私国家禁止进出口的货物、物品",\n    "走私普通货物、物品": "走私普通货物、物品",\n    "走私普通货物、物品罪": "走私普通货物、物品",\n    "走私武器、弹药": "走私武器、弹药",\n    "走私武器、弹药罪": "走私武器、弹药",\n    "走私罪": "走私",\n    "过失以危险方法危害公共安全": "过失以危险方法危害公共安全",\n    "过失以危险方法危害公共安全罪": "过失以危险方法危害公共安全",\n    "过失投放危险物质": "过失投放危险物质",\n    "过失投放危险物质罪": "过失投放危险物质",\n    "过失损坏武器装备、军事设施、军事通信": "过失损坏武器装备、军事设施、军事通信",\n    "过失损坏武器装备、军事设施、军事通信罪": "过失损坏武器装备、军事设施、军事通信",\n    "过失致人死亡": "过失致人死亡",\n    "过失致人死亡罪": "过失致人死亡",\n    "过失致人重伤": "过失致人重伤",\n    "过失致人重伤罪": "过失致人重伤",\n    "违法发放贷款": "违法发放贷款",\n    "违法发放贷款罪": "违法发放贷款",\n    "逃税": "逃税",\n    "逃税罪": "逃税",\n    "重大劳动安全事故": "重大劳动安全事故",\n    "重大劳动安全事故罪": "重大劳动安全事故",\n    "重大责任事故": "重大责任事故",\n    "重大责任事故罪": "重大责任事故",\n    "重婚": "重婚",\n    "重婚罪": "重婚",\n    "金融凭证诈骗": "金融凭证诈骗",\n    "金融凭证诈骗罪": "金融凭证诈骗",\n    "销售假冒注册商标的商品": "销售假冒注册商标的商品",\n    "销售假冒注册商标的商品罪": "销售假冒注册商标的商品",\n    "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告": "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告",\n    "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告罪": "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告",\n    "集资诈骗": "集资诈骗",\n    "集资诈骗罪": "集资诈骗",\n    "非国家工作人员受贿": "非国家工作人员受贿",\n    "非国家工作人员受贿罪": "非国家工作人员受贿",\n    "非法侵入住宅": "非法侵入住宅",\n    "非法侵入住宅罪": "非法侵入住宅",\n    "非法出售发票": "非法出售发票",\n    "非法出售发票罪": "非法出售发票",\n    "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物": "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物",\n    "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪": "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物",\n    "非法制造、销售非法制造的注册商标标识": "非法制造、销售非法制造的注册商标标识",\n    "非法制造、销售非法制造的注册商标标识罪": "非法制造、销售非法制造的注册商标标识",\n    "非法占用农用地": "非法占用农用地",\n    "非法占用农用地罪": "非法占用农用地",\n    "非法吸收公众存款": "非法吸收公众存款",\n    "非法吸收公众存款罪": "非法吸收公众存款",\n    "非法拘禁": "非法拘禁",\n    "非法拘禁罪": "非法拘禁",\n    "非法持有、私藏枪支、弹药": "非法持有、私藏枪支、弹药",\n    "非法持有、私藏枪支、弹药罪": "非法持有、私藏枪支、弹药",\n    "非法持有毒品": "非法持有毒品",\n    "非法持有毒品罪": "非法持有毒品",\n    "非法捕捞水产品": "非法捕捞水产品",\n    "非法捕捞水产品罪": "非法捕捞水产品",\n    "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品": "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品",\n    "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品罪": "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品",\n    "非法收购、运输盗伐、滥伐的林木": "非法收购、运输盗伐、滥伐的林木",\n    "非法收购、运输盗伐、滥伐的林木罪": "非法收购、运输盗伐、滥伐的林木",\n    "非法狩猎": "非法狩猎",\n    "非法狩猎罪": "非法狩猎",\n    "非法猎捕、杀害珍贵、濒危野生动物": "非法猎捕、杀害珍贵、濒危野生动物",\n    "非法猎捕、杀害珍贵、濒危野生动物罪": "非法猎捕、杀害珍贵、濒危野生动物",\n    "非法生产、买卖警用装备": "非法生产、买卖警用装备",\n    "非法生产、买卖警用装备罪": "非法生产、买卖警用装备",\n    "非法生产、销售间谍专用器材": "非法生产、销售间谍专用器材",\n    "非法生产、销售间谍专用器材罪": "非法生产、销售间谍专用器材",\n    "非法种植毒品原植物": "非法种植毒品原植物",\n    "非法种植毒品原植物罪": "非法种植毒品原植物",\n    "非法组织卖血": "非法组织卖血",\n    "非法组织卖血罪": "非法组织卖血",\n    "非法经营": "非法经营",\n    "非法经营罪": "非法经营",\n    "非法获取公民个人信息": "非法获取公民个人信息",\n    "非法获取公民个人信息罪": "非法获取公民个人信息",\n    "非法获取国家秘密": "非法获取国家秘密",\n    "非法获取国家秘密罪": "非法获取国家秘密",\n    "非法行医": "非法行医",\n    "非法行医罪": "非法行医",\n    "非法进行节育手术": "非法进行节育手术",\n    "非法进行节育手术罪": "非法进行节育手术",\n    "非法采伐、毁坏国家重点保护植物": "非法采伐、毁坏国家重点保护植物",\n    "非法采伐、毁坏国家重点保护植物罪": "非法采伐、毁坏国家重点保护植物",\n    "非法采矿": "非法采矿",\n    "非法采矿罪": "非法采矿",\n    "骗取贷款、票据承兑、金融票证": "骗取贷款、票据承兑、金融票证",\n    "骗取贷款、票据承兑、金融票证罪": "骗取贷款、票据承兑、金融票证"\n  },\n  "dispute_focus_identification": {\n    "原审判决是否适当": "原审判决是否适当",\n    "合同效力": "合同效力",\n    "合同解除": "合同解除",\n    "损失认定及处理": "损失认定及处理",\n    "本金争议": "本金争议",\n    "租金情况": "租金情况",\n    "诉讼主体": "诉讼主体",\n    "诉讼时效": "诉讼时效",\n    "财产分割": "财产分割",\n    "责任划分": "责任划分",\n    "责任承担": "责任承担",\n    "责任认定": "责任认定",\n    "违约": "违约",\n    "鉴定结论采信问题": "鉴定结论采信问题"\n  },\n  "legal_help_issue_identification": {\n    "benefits": "benefits",\n    "business": "business",\n    "consumer": "consumer",\n    "consumer law": "consumer",\n    "courts": "courts",\n    "crime": "crime",\n    "criminal law": "crime",\n    "divorce": "divorce",\n    "domestic_violence": "domestic_violence",\n    "education": "education",\n    "employment": "employment",\n    "employment law": "employment",\n    "estates": "estates",\n    "family": "family",\n    "family law": "family",\n    "health": "health",\n    "health law": "health",\n    "housing": "housing",\n    "housing law": "housing",\n    "immigration": "immigration",\n    "labor law": "employment",\n    "labour law": "employment",\n    "landlord-tenant law": "housing",\n    "tort law": "torts",\n    "torts": "torts",\n    "traffic": "traffic",\n    "traffic law": "traffic"\n  },\n  "legal_issue_identification": {\n    "交通事故": "交通事故",\n    "人身损害": "人身损害",\n    "侵权": "侵权",\n    "侵权纠纷": "侵权",\n    "债权债务": "债权债务",\n    "债权债务纠纷": "债权债务",\n    "公司法": "公司法",\n    "刑事辩护": "刑事辩护",\n    "劳动争议": "劳动纠纷",\n    "劳动纠纷": "劳动纠纷",\n    "医疗纠纷": "医疗纠纷",\n    "合同纠纷": "合同纠纷",\n    "婚姻家庭": "婚姻家庭",\n    "建设工程": "建设工程",\n    "房产纠纷": "房产纠纷",\n    "抵押担保": "抵押担保",\n    "拆迁安置": "拆迁安置",\n    "海事海商": "海事海商",\n    "消费权益": "消费权益",\n    "涉外法律": "涉外法律",\n    "知识产权": "知识产权",\n    "行政诉讼": "行政诉讼"\n  },\n  "oral_argument_purpose_identification": {\n    "background": "background",\n    "clarification": "clarification",\n    "communicate": "communicate",\n    "criticism": "criticism",\n    "humor": "humor",\n    "implications": "implications",\n    "support": "support"\n  },\n  "terms_clause_type": {\n    "access/edit/deletion": "access/edit/deletion",\n    "accountability": "accountability",\n    "accurate at another materiality standard (e.g., hybrid standard)": "accurate at another materiality standard (e.g., hybrid standard)",\n    "accurate in all material respects": "accurate in all material respects",\n    "accurate in all respects": "accurate in all respects",\n    "accurate in all respects with de minimis exception": "accurate in all respects with de minimis exception",\n    "actual knowledge": "actual knowledge",\n    "affiliate license-licensee": "affiliate license-licensee",\n    "affiliate license-licensor": "affiliate license-licensor",\n    "after signing only": "after signing only",\n    "all cash": "all cash",\n    "all mae carveouts": "all mae carveouts",\n    "all stock": "all stock",\n    "anti-assignment": "anti-assignment",\n    "applies only to specified negative covenants": "applies only to specified negative covenants",\n    "applies to all negative covenants": "applies to all negative covenants",\n    "arbitration": "arbitration",\n    "audit rights": "audit rights",\n    "audits": "audits",\n    "before or after signing": "before or after signing",\n    "cap on liability": "cap on liability",\n    "certification": "certification",\n    "change of control": "change of control",\n    "choice of law": "governing law",\n    "choice of law clause": "governing law",\n    "collection and use": "collection/use",\n    "collection/use": "collection/use",\n    "competitive restriction exception": "competitive restriction exception",\n    "confidentiality of the agreement": "confidentiality of the agreement",\n    "constructive knowledge": "constructive knowledge",\n    "content removal": "content removal",\n    "continuing confidentiality after termination": "continuing confidentiality after termination",\n    "contract by using": "contract by using",\n    "covenant not to sue": "covenant not to sue",\n    "data collection and use": "collection/use",\n    "data retention": "retention",\n    "data security": "security",\n    "effective date": "effective date",\n    "exclusivity": "exclusivity",\n    "expiration date": "expiration date",\n    "express identification of protected information": "express identification of protected information",\n    "external disclosure permission": "external disclosure permission",\n    "first-party and third-party collection": "first-party and third-party collection",\n    "first-party collection": "first-party collection",\n    "flat covenant (no efforts standard)": "flat covenant (no efforts standard)",\n    "governing law": "governing law",\n    "inconsistent\\" with fiduciary duties": "inconsistent\\" with fiduciary duties",\n    "independent acquisition exception": "independent acquisition exception",\n    "independent development exception": "independent development exception",\n    "insurance": "insurance",\n    "internal personnel disclosure permission": "internal personnel disclosure permission",\n    "ip ownership assignment": "ip ownership assignment",\n    "irrevocable or perpetual license": "irrevocable or perpetual license",\n    "joint ip ownership": "joint ip ownership",\n    "jurisdiction": "jurisdiction",\n    "known, but consequences unknown, at signing": "known, but consequences unknown, at signing",\n    "liability cap": "cap on liability",\n    "license grant": "license grant",\n    "limitation of liability": "limitation of liability",\n    "liquidated damages": "liquidated damages",\n    "minimum commitment": "minimum commitment",\n    "mixed cash/stock": "mixed cash/stock",\n    "mixed cash/stock: election": "mixed cash/stock: election",\n    "most favored nation": "most favored nation",\n    "no grant of intellectual-property rights": "no grant of intellectual-property rights",\n    "no-solicit of customers": "no-solicit of customers",\n    "no-solicit of employees": "no-solicit of employees",\n    "non-compete": "non-compete",\n    "non-disparagement": "non-disparagement",\n    "non-transferable license": "non-transferable license",\n    "not known and not reasonably foreseeable at signing": "not known and not reasonably foreseeable at signing",\n    "not known at signing": "not known at signing",\n    "notice before compelled disclosure": "notice before compelled disclosure",\n    "notice period to terminate renewal": "notice period to terminate renewal",\n    "permitted copying of protected information": "permitted copying of protected information",\n    "permitted retention after termination": "permitted retention after termination",\n    "policy amendment": "policy amendment",\n    "post-termination services": "post-termination services",\n    "price restrictions": "price restrictions",\n    "protection of oral disclosures": "protection of oral disclosures",\n    "purpose-limited use of confidential information": "purpose-limited use of confidential information",\n    "reasonable standard": "reasonable standard",\n    "reasonableness limitation": "reasonableness limitation",\n    "reasonably expected fiduciary inconsistency from failure to act": "reasonably expected fiduciary inconsistency from failure to act",\n    "renewal term": "renewal term",\n    "retention": "retention",\n    "return or destruction of protected information": "return or destruction of protected information",\n    "revenue-profit sharing": "revenue-profit sharing",\n    "right to audit": "audit rights",\n    "rofr-rofo-rofn": "rofr-rofo-rofn",\n    "security": "security",\n    "some mae carveouts": "some mae carveouts",\n    "source code escrow": "source code escrow",\n    "specific audiences": "specific audiences",\n    "strict liability": "strict liability",\n    "termination for convenience": "termination for convenience",\n    "third party beneficiary": "third party beneficiary",\n    "third-party collection": "third-party collection",\n    "third-party data sharing": "third-party sharing",\n    "third-party sharing": "third-party sharing",\n    "tracking preference": "tracking preference",\n    "training": "training",\n    "uncapped liability": "uncapped liability",\n    "unilateral change": "unilateral change",\n    "unilateral termination": "unilateral termination",\n    "unlimited-all-you-can-eat-license": "unlimited-all-you-can-eat-license",\n    "user choice": "user choice",\n    "verification": "verification",\n    "volume restriction": "volume restriction",\n    "warranty duration": "warranty duration",\n    "would\\" (reasonably) be expected to": "would\\" (reasonably) be expected to"\n  },\n  "trademark_distinctiveness": {\n    "arbitrary": "arbitrary",\n    "descriptive": "descriptive",\n    "fanciful": "fanciful",\n    "generic": "generic",\n    "suggestive": "suggestive"\n  }\n}\n'
POLICY_BYTES = b'{\n  "version": "legal_shortqa_em_v2",\n  "purpose": "Deterministic complete-answer equivalence, not fuzzy semantic judging",\n  "oral_intents": {\n    "background": ["background", "providing background", "background information", "\xe8\x83\x8c\xe6\x99\xaf\xe4\xbb\x8b\xe7\xbb\x8d"],\n    "clarification": ["clarification", "seeking clarification", "request for clarification", "asking for clarification", "clarification request", "\xe6\xbe\x84\xe6\xb8\x85", "\xe8\xaf\xb7\xe6\xb1\x82\xe6\xbe\x84\xe6\xb8\x85"],\n    "communicate": ["communicate", "communication", "communicating information", "\xe4\xbc\xa0\xe8\xbe\xbe\xe4\xbf\xa1\xe6\x81\xaf"],\n    "support": ["support", "expressing support", "\xe6\x94\xaf\xe6\x8c\x81"],\n    "criticism": ["criticism", "expressing criticism", "expressing skepticism", "\xe6\x89\xb9\xe8\xaf\x84", "\xe8\xb4\xa8\xe7\x96\x91"],\n    "implications": ["implications", "exploring implications", "\xe6\x8e\xa2\xe7\xa9\xb6\xe5\xbd\xb1\xe5\x93\x8d"],\n    "humor": ["humor", "humour", "humorous remark", "\xe5\xb9\xbd\xe9\xbb\x98"]\n  },\n  "orthographic_word_aliases": {"licence": "license", "nonexclusive": "non-exclusive"},\n  "clause_head_aliases": {"clause": "clause", "provision": "clause"},\n  "pest_duty_holders": {"landlord": "landlord", "\xe6\x88\xbf\xe4\xb8\x9c": "landlord", "\xe5\x87\xba\xe7\xa7\x9f\xe4\xba\xba": "landlord", "\xe5\x87\xba\xe7\xa7\x9f\xe6\x96\xb9": "landlord", "tenant": "tenant", "\xe7\xa7\x9f\xe6\x88\xb7": "tenant", "\xe6\x89\xbf\xe7\xa7\x9f\xe4\xba\xba": "tenant"},\n  "constraints": [\n    "No substring, token-overlap, edit-distance, stemming or embedding credit",\n    "No removal of negation, exclusivity, duty holder, legal version, article suffix, paragraph or item",\n    "Additional/missing required labels must fail; complete equivalent golds remain alternatives, not a bag of independent answers",\n    "Lease-basis phrases may be omitted only when the task question explicitly fixes that same basis",\n    "Rules are task-scoped and do not assert correctness of source golds"\n  ]\n}\n'
assert hashlib.sha256(VOCAB_JSON.encode()).hexdigest() == VOCAB_SHA256
POLICY_SHA256 = hashlib.sha256(POLICY_BYTES).hexdigest()
assert POLICY_SHA256 == '2ea8662d17fe5f1d1af8ccce21633474724487ab5e3ce8c76dd65081a0668502'
POLICY = json.loads(POLICY_BYTES)
VERSION = 'legal_shortqa_em_v2'

surface=v1_surface
N=r'[0-9零〇一二三四五六七八九十百千]+'
ORD=r'\(?('+N+r')\)?'
LAW=r'(?:(?:(?P<year>[12][0-9]{3})年)?(?P<law>中华人民共和国刑法|中华人民共和国民法典|刑法|民法典))'
CITATION=re.compile(
    r'(?P<lawpart>'+LAW+r')?第(?P<articles>'+N+r'(?:、'+N+r')*)条'
    r'(?:之(?P<suffix>'+N+r'))?'
    r'(?:第(?P<paragraph>'+N+r'|\('+N+r'\))款)?'
    r'(?:第(?P<item>'+N+r'|\('+N+r'\))项)?'
    r'(?P<annotation>\([^()]{1,40}\))?')
CONTINUATION=re.compile(r'第(?P<number>'+N+r'|\('+N+r'\))(?P<kind>款|项)')
SEPARATOR=re.compile(r'[\s、,;和及与]*')

def number(s):
    if s.startswith('(') and s.endswith(')'):s=s[1:-1]
    if s.isdigit():return int(s)
    s=s.replace('〇','零')
    digits={c:i for i,c in enumerate('零一二三四五六七八九')}
    if all(c in digits for c in s):return int(''.join(str(digits[c]) for c in s))
    total=current=0;last_unit=10000
    for c in s:
        if c in digits:current=digits[c]
        elif c in '十百千':
            unit={'十':10,'百':100,'千':1000}[c]
            if unit>=last_unit:raise ValueError('Non-decreasing Chinese numeral units')
            total+=(current or 1)*unit;current=0;last_unit=unit
        else:raise ValueError('Invalid legal ordinal')
    return total+current

def articles(text):
    """Retain law/version/article/suffix/paragraph/item and any explicit annotation."""
    s=surface(text).replace('《','').replace('》','')
    pos=0;law=None;year=None;previous=None;out=[]
    while pos<len(s):
        gap=SEPARATOR.match(s,pos);pos=gap.end()
        if pos==len(s):break
        if out and s[pos:] in ('的规定','规定'):pos=len(s);break
        m=CITATION.match(s,pos)
        try:
            if m:
                if m['law']:
                    law='刑法' if '刑法' in m['law'] else '民法典';year=m['year']
                if not law:return None
                ns=[number(n) for n in m['articles'].split('、')]
                suffix=number(m['suffix']) if m['suffix'] else None
                paragraph=number(m['paragraph']) if m['paragraph'] else None
                item=number(m['item']) if m['item'] else None
                if any(n<=0 for n in ns+[x for x in (suffix,paragraph,item) if x is not None]):return None
                if suffix and len(ns)>1:return None  # Ambiguous suffix scope.
                for n in ns:
                    previous=(law,year,n,suffix,paragraph,item,m['annotation'])
                    out.append(previous)
            else:
                m=CONTINUATION.match(s,pos)
                if not m or not previous:return None
                n=number(m['number'])
                if n<=0:return None
                a=list(previous)
                if m['kind']=='款':a[4]=n;a[5]=None
                else:a[5]=n
                a[6]=None;previous=tuple(a);out.append(previous)
        except ValueError:return None
        pos=m.end()
    return tuple(sorted(set(out),key=str)) if out else None

def crime_parts(text):
    # Normalize fullwidth semicolons BEFORE splitting. Internal punctuation in
    # compound statutory charge names must not be treated as independent crimes.
    return [a.strip() for a in re.split(r';|(?<=罪)[、,和及与]',surface(text)) if a.strip()]

def local_vocab(task,refs,vocab):
    d={k:v for k,v in vocab.get(task,{}).items() if task!='criminal_charge_judgment' or ';' not in surface(k)}
    for ref in refs:
        if not isinstance(ref,str):continue
        for atom in crime_parts(ref) if task=='criminal_charge_judgment' else surface(ref).split(';'):
            key=surface(atom)
            if task=='criminal_charge_judgment':
                base=key[:-1] if key.endswith('罪') else key
                # Preserve known aliases; unknown held-out atomic names can
                # normalize only their own suffix, never another row's label.
                target=d.get(base,d.get(base+'罪',base))
                if target.endswith('罪'):target=target[:-1]
                d[base]=target;d[base+'罪']=target
            else:d.setdefault(key,key)
    return d

def basis_is_fixed(info):
    contract=info.get('legal_answer_contract',{})
    if not isinstance(contract,dict) or contract.get('version')!='legal_answer_contract_v2':return False
    q=contract.get('question','')
    # Only a frozen, explicit question is used; never model output or other rows.
    return isinstance(q,str) and bool(re.search(r'\blease\b|租约|租赁合同',q,re.I))

def open_answer(text,task,info):
    s=surface(text)
    for a,b in POLICY['orthographic_word_aliases'].items():
        s=re.sub(r'\b'+re.escape(a)+r'\b',b,s)
    if task not in {'terms_clause_type','legal_help_issue_identification','legal_issue_identification','dispute_focus_identification'}:
        return ('surface',s)
    # Full-string grammar, not deletion of arbitrary prepositions/qualifiers.
    if task=='terms_clause_type':
        m=re.fullmatch(r'(?:the )?definition of ([a-z][a-z0-9 -]{0,80})',s)
        if m:return ('definition',m[1])
        m=re.fullmatch(r'([a-z][a-z0-9 -]{0,80}) definition',s)
        if m:return ('definition',m[1])
        m=re.fullmatch(r'(?:a |the )?([a-z][a-z -]{0,70}) (clause|provision)',s)
        if m:return ('clause',m[1])
        m=re.fullmatch(r'(?:a |an |the )?((?:non-exclusive|exclusive)(?: [a-z-]+){0,6} (?:license))',s)
        if m:return ('surface',m[1])
        if re.fullmatch(r'(?:non-exclusive|exclusive)(?: [a-z-]+){0,6} license',s):return ('surface',s)
    holder=None;lease=False
    m=re.fullmatch(r'(?:(?:under|pursuant to) the lease,? )?(?:the )?(landlord|tenant)(?:\x27s)? '
        r'(?:(?:responsibility|obligation|duty) (?:for|to provide) pest control|pest[- ]control (?:responsibility|obligation|duty))'
        r'( under the lease)?',s)
    if m:holder=m[1];lease=bool(m[2]) or s.startswith(('under the lease','pursuant to the lease'))
    else:
        m=re.fullmatch(r'(?P<basis>(?:租约|租赁合同)(?:规定|约定)的)?(?P<holder>房东|出租人|出租方|租户|承租人)的?'
            r'(?:虫害防治|害虫防治|除虫|灭虫)(?:义务|责任)',s)
        if m:holder=POLICY['pest_duty_holders'][m['holder']];lease=bool(m['basis'])
    if holder:
        return ('pest_control_duty',holder,'fixed_question_basis' if basis_is_fixed(info) else ('lease' if lease else 'unspecified'))
    return ('surface',s)

def canonical(text,task,vocab,info,refs):
    if not isinstance(text,str) or not text.strip():return None
    if task=='statute_article_grounding':return articles(text)
    if task=='criminal_charge_judgment':return v1_labels(text,local_vocab(task,refs,vocab))
    if task=='oral_argument_purpose_identification':
        d={surface(a):label for label,aliases in POLICY['oral_intents'].items() for a in aliases}
        return v1_labels(text,d)
    # Preserve v1 closed-label equivalences for other tasks. Open phrases may
    # use the finite full-string rules below, but never label substrings.
    return open_answer(text,task,info)

def compute_score(solution_str,ground_truth,extra_info=None):
    info=extra_info or {}
    if info.get('legal_em_version')!=VERSION:raise ValueError('Legal v2 version missing or mismatched')
    if info.get('legal_normalization_policy_sha256')!=POLICY_SHA256:raise ValueError('Legal v2 policy missing or mismatched')
    vocab=v1_vocabulary(info['legal_vocab_sha256']);task=info['legal_task']
    refs=list(ground_truth) if isinstance(ground_truth,(list,tuple)) else [ground_truth]
    if not isinstance(solution_str,str):return 0.
    pred=canonical(solution_str,task,vocab,info,refs)
    for ref in refs:
        if not isinstance(ref,str):continue
        if solution_str.strip()==ref.strip():return 1.
        target=canonical(ref,task,vocab,info,refs)
        if pred is not None and target is not None and pred==target:return 1.
    # Existing complete-label mappings are retained as a second exact path.
    if task not in {'statute_article_grounding','criminal_charge_judgment','oral_argument_purpose_identification'}:
        v1info={**info,'legal_em_version':v1_VERSION}
        return v1_compute_score(solution_str,refs,v1info)
    return 0.

SYSTEM_PROMPT = "Let's think step by step and output the final answer within <answer> </answer> tags. Keep only the requested short answer inside the tags. Then output your confidence on a separate line: <conf> Confidence: <float value> where the value is between 0 and 1."


def validate_info(info):
    if info.get('legal_em_version') != VERSION:
        raise ValueError('Expected the frozen legal_shortqa_em_v2 metadata')
    if info.get('legal_normalization_policy_sha256') != POLICY_SHA256:
        raise ValueError('Wrong or missing legal normalization policy hash')
    if info.get('legal_vocab_sha256') != VOCAB_SHA256:
        raise ValueError('Wrong or missing frozen legal vocabulary hash')
    if not info.get('legal_task'):
        raise ValueError('Missing legal_task')


def convert_rows(rows):
    converted=[]
    for row in rows:
        info=copy.deepcopy(row['extra_info'])
        validate_info(info)
        if row['data_source'] != 'LegalNormalizedEM-v2':
            raise ValueError('Refusing to silently upgrade a different dataset/scorer')
        if info.get('parse_mode') != 'answer_tag' or not info.get('context_provided'):
            raise ValueError('Expected the frozen with-context answer_tag protocol')
        content=info['model_material']+'\n\nQuestion: '+info['model_instruction']
        golds=row['gold_answers']
        if row['question'] != content or row['prompt'] != content:
            raise ValueError('JSONL model-facing content does not match frozen metadata')
        if not golds or row['reward_model']['ground_truth'] != golds:
            raise ValueError('Missing or inconsistent full gold list')
        if hashlib.sha256(info['model_material'].encode()).hexdigest() != info['model_material_sha256']:
            raise ValueError('Material hash mismatch')
        converted.append({'data_source':row['data_source'],
            'prompt':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':content}],
            'ability':row['ability'],'reward_model':copy.deepcopy(row['reward_model']),
            'extra_info':info})
    if not converted:
        raise ValueError('Empty dataset; if this is an LFS pointer, run git lfs pull')
    return converted


def prepare_jsonl(path, output_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    with path.open(encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    expected=convert_rows(rows)
    dest=output_dir/'legal_input.parquet'
    if dest.exists():
        if pq.read_table(dest).to_pylist() != expected:
            raise ValueError('Existing prepared input differs; choose a fresh output directory')
    else:
        pq.write_table(pa.Table.from_pylist(expected),dest)
    return dest, len(rows)


def self_test():
    def info(task,question=''):
        return {'legal_task':task,'legal_em_version':VERSION,'legal_vocab_sha256':VOCAB_SHA256,
            'legal_normalization_policy_sha256':POLICY_SHA256,
            'legal_answer_contract':{'version':'legal_answer_contract_v2','question':question}}
    cases=[
        ('legal_help_issue_identification','Child custody','child custody','',1),
        ('statute_article_grounding','中华人民共和国刑法第二百七十六条之一','刑法第276条之一','',1),
        ('statute_article_grounding','刑法第276条','刑法第276条之一','',0),
        ('statute_article_grounding','刑法第263条第1项;刑法第263条第5项','刑法第263条第（一）项、第（五）项','',1),
        ('criminal_charge_judgment','贪污罪；职务侵占罪','职务侵占；贪污','',1),
        ('criminal_charge_judgment','贪污罪','职务侵占；贪污','',0),
        ('oral_argument_purpose_identification','Seeking clarification','Clarification','',1),
        ('oral_argument_purpose_identification','not seeking clarification','Clarification','',0),
        ('terms_clause_type','A nonexclusive licence','Non-exclusive license','',1),
        ('terms_clause_type','Exclusive license','Non-exclusive license','',0),
        ('legal_help_issue_identification',"Landlord's responsibility for pest control under the lease",'landlord pest control obligation','What lease obligation concerns pest control?',1),
        ('legal_help_issue_identification',"Tenant's pest control obligation",'landlord pest control obligation','What lease obligation concerns pest control?',0),
    ]
    for task,pred,gold,question,expected in cases:
        assert compute_score(pred,[gold],info(task,question)) == expected,(task,pred,gold)
    assert json.loads(json.dumps({'text':'a\u2028b'},ensure_ascii=False)+'\n') == {'text':'a\u2028b'}
    try:validate_info({})
    except ValueError:pass
    else:raise AssertionError('Missing metadata must fail closed')
    print(json.dumps({'self_test':'passed','scoring_cases':len(cases),
        'scorer':VERSION,'policy_sha256':POLICY_SHA256,'vocabulary_sha256':VOCAB_SHA256},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--data-path',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args,rest=parser.parse_known_args()
    if args.self_test:
        if rest or args.data_path or args.output_dir or args.prepare_only:
            parser.error('--self-test must be used alone')
        self_test();return
    if '-h' in rest or '--help' in rest:
        print(__doc__);return
    if not args.data_path or not args.output_dir:
        parser.error('--data-path and --output-dir are required')
    if args.data_path.suffix.lower() != '.jsonl':
        parser.error('Use the original frozen JSONL, not a separately converted input')
    if any(a=='--correctness-reward' or a.startswith('--correctness-reward=') for a in rest):
        parser.error('This entry point pins legal EM v2; do not override correctness')
    if '--enable-thinking' in rest:
        parser.error('This protocol uses Qwen NoThink; do not enable thinking')
    defaults={'--temperature':'0.7','--top-p':'1.0','--seed':'42',
        '--max-tokens':'6144','--max-model-len':'24576','--eval-rollouts':'1',
        '--batch-size':'32','--max-num-seqs':'128','--max-num-batched-tokens':'8192',
        '--judge-concurrency':'8','--tensor-parallel-size':'2','--gpu-memory-utilization':'0.7'}
    supplied={a.split('=',1)[0] for a in rest if a.startswith('--')}
    for flag,value in defaults.items():
        if flag not in supplied:rest.extend([flag,value])
    if '--resume-completions' not in supplied:rest.append('--resume-completions')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not (args.output_dir/'legal_protocol.json').exists() and any(
            (args.output_dir/name).exists() for name in ['metrics.json','results.jsonl','.completion_records.jsonl']):
        raise ValueError('Unversioned existing outputs; choose a fresh output directory')
    dest,count=prepare_jsonl(args.data_path,args.output_dir)
    protocol={'scorer':VERSION,'policy_sha256':POLICY_SHA256,'vocabulary_sha256':VOCAB_SHA256,
        'input_jsonl_sha256':hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
        'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'num_prompts':count,'system_prompt':SYSTEM_PROMPT,'online_correctness_judge':False}
    marker=args.output_dir/'legal_protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Refusing to mix a different protocol with existing outputs')
    marker.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps({'prepared':str(dest),**protocol},ensure_ascii=False,indent=2));return
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import evaluate_rlcr
    original=evaluate_rlcr.default_compute_score
    def dispatch(data_source,solution_str,ground_truth,extra_info=None,**kwargs):
        if data_source == 'LegalNormalizedEM-v2':
            validate_info(extra_info or {})
            return compute_score(solution_str,ground_truth,extra_info)
        return original(data_source=data_source,solution_str=solution_str,
            ground_truth=ground_truth,extra_info=extra_info,**kwargs)
    evaluate_rlcr.default_compute_score=dispatch
    import evaluate_baselines
    sys.argv=[sys.argv[0],'--data-path',str(dest),'--output-dir',str(args.output_dir),*rest]
    evaluate_baselines.main()


if __name__=='__main__':main()
