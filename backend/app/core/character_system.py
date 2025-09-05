"""
人设系统

实现AI角色的人格设定、一致性检查和动态调整功能
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from ..config import get_settings
from ..models.chat_models import ChatMessage, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CharacterSystem:
    """角色人设系统"""
    
    def __init__(self):
        self.character_profiles = {}
        self.consistency_cache = {}
        self._load_default_characters()
    
    def _load_default_characters(self):
        """加载默认角色配置"""
        # 默认助手角色
        default_character = {
            "id": "default",
            "name": "小助手",
            "description": "友善、专业、有帮助的AI助手",
            "personality": {
                "traits": ["友善", "耐心", "专业", "幽默"],
                "introverted": False,
                "formality_level": 0.6,
                "empathy_level": 0.8,
                "creativity_level": 0.7,
                "assertiveness": 0.6
            },
            "knowledge_domains": ["通用知识", "技术支持", "日常对话", "学习辅导"],
            "writing_style": {
                "tone": "温和友善",
                "sentence_length": "中等",
                "use_emoji": True,
                "technical_level": "适中",
                "formality": "半正式",
                "humor_style": "温和幽默"
            },
            "behavioral_patterns": {
                "greeting_style": "热情但不过分",
                "question_handling": "耐心详细",
                "error_response": "承认错误并提供帮助",
                "farewell_style": "温暖友善"
            },
            "constraints": {
                "forbidden_topics": ["政治敏感", "暴力内容", "成人内容"],
                "response_length": {"min": 10, "max": 500},
                "language_preference": "简体中文"
            }
        }
        
        # 专业顾问角色
        professional_character = {
            "id": "professional",
            "name": "专业顾问",
            "description": "专业、严谨、高效的商务顾问",
            "personality": {
                "traits": ["专业", "严谨", "高效", "理性"],
                "introverted": True,
                "formality_level": 0.9,
                "empathy_level": 0.6,
                "creativity_level": 0.5,
                "assertiveness": 0.8
            },
            "knowledge_domains": ["商务咨询", "项目管理", "数据分析", "战略规划"],
            "writing_style": {
                "tone": "专业严谨",
                "sentence_length": "中长",
                "use_emoji": False,
                "technical_level": "高",
                "formality": "正式",
                "humor_style": "职场幽默"
            },
            "behavioral_patterns": {
                "greeting_style": "简洁专业",
                "question_handling": "结构化回答",
                "error_response": "承认并提供解决方案",
                "farewell_style": "专业礼貌"
            }
        }
        
        # 活泼助手角色
        cheerful_character = {
            "id": "cheerful",
            "name": "小可爱",
            "description": "活泼可爱、充满活力的年轻助手",
            "personality": {
                "traits": ["活泼", "可爱", "热情", "乐观"],
                "introverted": False,
                "formality_level": 0.3,
                "empathy_level": 0.9,
                "creativity_level": 0.9,
                "assertiveness": 0.5
            },
            "knowledge_domains": ["日常聊天", "娱乐资讯", "生活小贴士", "情感支持"],
            "writing_style": {
                "tone": "活泼可爱",
                "sentence_length": "短句为主",
                "use_emoji": True,
                "technical_level": "简单",
                "formality": "非正式",
                "humor_style": "俏皮幽默"
            },
            "behavioral_patterns": {
                "greeting_style": "热情活泼",
                "question_handling": "耐心引导",
                "error_response": "可爱承认错误",
                "farewell_style": "温馨告别"
            }
        }
        
        # 存储角色配置
        self.character_profiles = {
            "default": default_character,
            "professional": professional_character,
            "cheerful": cheerful_character
        }
        
        logger.info(f"加载了 {len(self.character_profiles)} 个默认角色")
    
    def get_character(self, character_id: str = "default") -> Dict[str, Any]:
        """获取角色配置"""
        return self.character_profiles.get(character_id, self.character_profiles["default"])
    
    def list_characters(self) -> List[Dict[str, Any]]:
        """列出所有可用角色"""
        return [
            {
                "id": char_id,
                "name": char_config["name"],
                "description": char_config["description"]
            }
            for char_id, char_config in self.character_profiles.items()
        ]
    
    async def apply_character_consistency(
        self,
        response: str,
        character_config: Dict[str, Any],
        conversation_context: Optional[List[ChatMessage]] = None
    ) -> Tuple[str, bool]:
        """应用角色一致性检查和调整"""
        try:
            if not settings.character.consistency.check_enabled:
                return response, False
            
            # 检查各个维度的一致性
            consistency_issues = []
            
            # 1. 性格一致性检查
            personality_issues = await self._check_personality_consistency(
                response, character_config["personality"]
            )
            consistency_issues.extend(personality_issues)
            
            # 2. 写作风格一致性检查
            style_issues = await self._check_writing_style_consistency(
                response, character_config["writing_style"]
            )
            consistency_issues.extend(style_issues)
            
            # 3. 知识领域一致性检查
            domain_issues = await self._check_knowledge_domain_consistency(
                response, character_config["knowledge_domains"]
            )
            consistency_issues.extend(domain_issues)
            
            # 4. 行为模式一致性检查
            behavior_issues = await self._check_behavioral_consistency(
                response, character_config["behavioral_patterns"], conversation_context
            )
            consistency_issues.extend(behavior_issues)
            
            # 如果发现问题，尝试调整
            if consistency_issues:
                logger.info(f"发现一致性问题: {', '.join(consistency_issues)}")
                
                adjusted_response = await self._adjust_response_for_consistency(
                    response, character_config, consistency_issues
                )
                
                return adjusted_response, True
            
            return response, False
            
        except Exception as e:
            logger.error(f"角色一致性检查失败: {e}")
            return response, False
    
    async def _check_personality_consistency(
        self,
        response: str,
        personality: Dict[str, Any]
    ) -> List[str]:
        """检查性格一致性"""
        issues = []
        response_lower = response.lower()
        
        # 内向/外向检查
        if personality.get("introverted", False):
            # 内向角色应该更谨慎、简洁
            extroverted_indicators = [
                "太棒了", "超级", "非常兴奋", "迫不及待", "大声"
            ]
            if any(indicator in response_lower for indicator in extroverted_indicators):
                issues.append("表现过于外向")
        else:
            # 外向角色应该更热情、活泼
            if len(response) < 20 and not any(punct in response for punct in ["!", "？", "~"]):
                issues.append("表现过于内向")
        
        # 正式程度检查
        formality_level = personality.get("formality_level", 0.5)
        
        if formality_level > 0.7:
            # 高正式度，不应该使用非正式用语
            informal_words = ["哈哈", "嘿", "咋", "啥", "咋样", "整"]
            if any(word in response_lower for word in informal_words):
                issues.append("用语过于随意")
        elif formality_level < 0.4:
            # 低正式度，不应该过于正式
            formal_words = ["敬请", "恭敬", "谨此", "特此", "恳请"]
            if any(word in response_lower for word in formal_words):
                issues.append("用语过于正式")
        
        # 共情能力检查
        empathy_level = personality.get("empathy_level", 0.5)
        if empathy_level > 0.7:
            # 高共情角色应该表现出关心和理解
            user_emotion_keywords = ["难过", "困难", "问题", "担心", "焦虑"]
            if any(keyword in response_lower for keyword in user_emotion_keywords):
                empathy_words = ["理解", "感受", "支持", "帮助", "陪伴"]
                if not any(word in response_lower for word in empathy_words):
                    issues.append("缺乏共情表达")
        
        return issues
    
    async def _check_writing_style_consistency(
        self,
        response: str,
        writing_style: Dict[str, Any]
    ) -> List[str]:
        """检查写作风格一致性"""
        issues = []
        
        # 表情符号使用检查
        use_emoji = writing_style.get("use_emoji", True)
        emoji_pattern = r'[😀-🙏]|[🌀-🗿]|[🚀-🛿]|[🇦-🇿]'
        has_emoji = bool(re.search(emoji_pattern, response))
        
        if use_emoji and not has_emoji and len(response) > 50:
            issues.append("缺少表情符号")
        elif not use_emoji and has_emoji:
            issues.append("不应使用表情符号")
        
        # 句子长度检查
        sentence_length = writing_style.get("sentence_length", "中等")
        sentences = re.split(r'[。！？.!?]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_length = sum(len(s) for s in sentences) / len(sentences)
            
            if sentence_length == "短句" and avg_length > 20:
                issues.append("句子过长")
            elif sentence_length == "长句" and avg_length < 15:
                issues.append("句子过短")
        
        # 语调检查
        tone = writing_style.get("tone", "中性")
        if tone == "温和友善":
            harsh_words = ["必须", "应该", "错误", "不对", "不行"]
            if any(word in response for word in harsh_words):
                issues.append("语调过于严厉")
        elif tone == "专业严谨":
            casual_words = ["哈哈", "呀", "哦", "嗯嗯", "好吧"]
            if any(word in response for word in casual_words):
                issues.append("语调过于随意")
        
        return issues
    
    async def _check_knowledge_domain_consistency(
        self,
        response: str,
        knowledge_domains: List[str]
    ) -> List[str]:
        """检查知识领域一致性"""
        issues = []
        
        # 定义各领域的关键词
        domain_keywords = {
            "技术支持": ["代码", "编程", "算法", "数据库", "API", "框架", "系统"],
            "商务咨询": ["战略", "市场", "客户", "业务", "营收", "成本", "ROI"],
            "日常对话": ["生活", "朋友", "家庭", "心情", "天气", "美食", "电影"],
            "学习辅导": ["学习", "课程", "考试", "知识点", "练习", "复习", "理解"],
            "娱乐资讯": ["电影", "音乐", "游戏", "明星", "综艺", "动漫", "小说"],
            "医疗健康": ["健康", "症状", "治疗", "药物", "医生", "医院", "检查"],
            "法律咨询": ["法律", "合同", "权利", "义务", "法规", "诉讼", "律师"]
        }
        
        # 检查是否涉及不在知识领域内的专业话题
        response_lower = response.lower()
        for domain, keywords in domain_keywords.items():
            if domain not in knowledge_domains:
                keyword_count = sum(1 for keyword in keywords if keyword in response_lower)
                if keyword_count >= 3:  # 如果涉及多个该领域关键词
                    issues.append(f"涉及超出知识范围的{domain}话题")
        
        return issues
    
    async def _check_behavioral_consistency(
        self,
        response: str,
        behavioral_patterns: Dict[str, Any],
        conversation_context: Optional[List[ChatMessage]] = None
    ) -> List[str]:
        """检查行为模式一致性"""
        issues = []
        
        if not conversation_context:
            return issues
        
        # 判断对话阶段
        is_greeting = len(conversation_context) <= 2
        is_farewell = any("再见" in msg.content or "拜拜" in msg.content 
                         for msg in conversation_context[-3:] if msg.role == MessageRole.USER)
        
        # 问候风格检查
        if is_greeting:
            greeting_style = behavioral_patterns.get("greeting_style", "标准")
            if greeting_style == "热情活泼":
                if not any(word in response for word in ["你好", "欢迎", "很高兴", "！"]):
                    issues.append("问候不够热情")
            elif greeting_style == "简洁专业":
                if len(response) > 50 or "！" in response:
                    issues.append("问候过于热情")
        
        # 告别风格检查
        if is_farewell:
            farewell_style = behavioral_patterns.get("farewell_style", "标准")
            if farewell_style == "温暖友善":
                if not any(word in response for word in ["再见", "保重", "祝", "期待"]):
                    issues.append("告别不够温暖")
        
        return issues
    
    async def _adjust_response_for_consistency(
        self,
        response: str,
        character_config: Dict[str, Any],
        issues: List[str]
    ) -> str:
        """根据一致性问题调整回复"""
        try:
            adjusted_response = response
            
            # 根据具体问题进行调整
            for issue in issues:
                if "表现过于外向" in issue:
                    # 降低外向表达
                    adjusted_response = re.sub(r'[!！]{2,}', '!', adjusted_response)
                    adjusted_response = adjusted_response.replace("太棒了", "很好")
                    adjusted_response = adjusted_response.replace("超级", "很")
                
                elif "表现过于内向" in issue:
                    # 增加外向表达
                    if not adjusted_response.endswith(('!', '！', '~')):
                        adjusted_response += "!"
                
                elif "用语过于随意" in issue:
                    # 提高正式程度
                    casual_formal_map = {
                        "哈哈": "",
                        "嘿": "您好",
                        "咋": "怎么",
                        "啥": "什么",
                        "整": "处理"
                    }
                    for casual, formal in casual_formal_map.items():
                        adjusted_response = adjusted_response.replace(casual, formal)
                
                elif "用语过于正式" in issue:
                    # 降低正式程度
                    formal_casual_map = {
                        "敬请": "请",
                        "恭敬": "",
                        "谨此": "",
                        "特此": "",
                        "恳请": "希望"
                    }
                    for formal, casual in formal_casual_map.items():
                        adjusted_response = adjusted_response.replace(formal, casual)
                
                elif "缺少表情符号" in issue:
                    # 添加适当的表情符号
                    writing_style = character_config.get("writing_style", {})
                    if writing_style.get("tone") == "活泼可爱":
                        adjusted_response += " 😊"
                    elif "帮助" in adjusted_response or "支持" in adjusted_response:
                        adjusted_response += " 💪"
                
                elif "不应使用表情符号" in issue:
                    # 移除表情符号
                    emoji_pattern = r'[😀-🙏]|[🌀-🗿]|[🚀-🛿]|[🇦-🇿]'
                    adjusted_response = re.sub(emoji_pattern, '', adjusted_response)
                
                elif "语调过于严厉" in issue:
                    # 软化语调
                    harsh_soft_map = {
                        "必须": "建议",
                        "应该": "可以",
                        "错误": "不太对",
                        "不对": "可能有问题",
                        "不行": "不太合适"
                    }
                    for harsh, soft in harsh_soft_map.items():
                        adjusted_response = adjusted_response.replace(harsh, soft)
                
                elif "语调过于随意" in issue:
                    # 提高专业度
                    casual_professional_map = {
                        "哈哈": "",
                        "呀": "",
                        "哦": "好的",
                        "嗯嗯": "是的",
                        "好吧": "明白了"
                    }
                    for casual, professional in casual_professional_map.items():
                        adjusted_response = adjusted_response.replace(casual, professional)
            
            # 确保调整后的回复不为空
            if not adjusted_response.strip():
                adjusted_response = response
            
            logger.info(f"回复已调整以保持角色一致性")
            return adjusted_response.strip()
            
        except Exception as e:
            logger.error(f"调整回复失败: {e}")
            return response
    
    def generate_character_prompt(self, character_config: Dict[str, Any]) -> str:
        """生成角色提示词"""
        try:
            prompt_parts = []
            
            # 基本角色信息
            name = character_config.get("name", "AI助手")
            description = character_config.get("description", "")
            
            prompt_parts.append(f"你是{name}，{description}。")
            
            # 性格特征
            personality = character_config.get("personality", {})
            if personality.get("traits"):
                traits = "、".join(personality["traits"])
                prompt_parts.append(f"你的性格特点是{traits}。")
            
            # 写作风格
            writing_style = character_config.get("writing_style", {})
            style_parts = []
            
            if writing_style.get("tone"):
                style_parts.append(f"语调{writing_style['tone']}")
            
            if writing_style.get("formality"):
                style_parts.append(f"表达方式{writing_style['formality']}")
            
            if writing_style.get("use_emoji"):
                style_parts.append("适当使用表情符号")
            
            if style_parts:
                prompt_parts.append(f"你的表达风格是: {', '.join(style_parts)}。")
            
            # 知识领域
            knowledge_domains = character_config.get("knowledge_domains", [])
            if knowledge_domains:
                domains = "、".join(knowledge_domains)
                prompt_parts.append(f"你擅长{domains}等领域。")
            
            # 行为准则
            behavioral_patterns = character_config.get("behavioral_patterns", {})
            if behavioral_patterns:
                prompt_parts.append("请根据以下行为准则回复:")
                
                if behavioral_patterns.get("question_handling"):
                    prompt_parts.append(f"- 回答问题时: {behavioral_patterns['question_handling']}")
                
                if behavioral_patterns.get("error_response"):
                    prompt_parts.append(f"- 遇到错误时: {behavioral_patterns['error_response']}")
            
            # 约束条件
            constraints = character_config.get("constraints", {})
            if constraints:
                if constraints.get("response_length"):
                    length_info = constraints["response_length"]
                    prompt_parts.append(f"回复长度控制在{length_info.get('min', 10)}-{length_info.get('max', 500)}字符之间。")
                
                if constraints.get("forbidden_topics"):
                    forbidden = "、".join(constraints["forbidden_topics"])
                    prompt_parts.append(f"避免涉及{forbidden}等话题。")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"生成角色提示词失败: {e}")
            return f"你是{character_config.get('name', 'AI助手')}，请提供有帮助的回复。"
    
    async def adapt_character_based_on_feedback(
        self,
        character_id: str,
        user_feedback: Dict[str, Any]
    ) -> bool:
        """根据用户反馈动态调整角色"""
        try:
            if character_id not in self.character_profiles:
                return False
            
            character_config = self.character_profiles[character_id]
            
            # 根据反馈调整性格参数
            if "formality_preference" in user_feedback:
                # 调整正式程度
                current_level = character_config["personality"]["formality_level"]
                preference = user_feedback["formality_preference"]
                
                if preference == "more_formal" and current_level < 0.9:
                    character_config["personality"]["formality_level"] += 0.1
                elif preference == "less_formal" and current_level > 0.1:
                    character_config["personality"]["formality_level"] -= 0.1
            
            if "humor_preference" in user_feedback:
                # 调整幽默风格
                humor_style = user_feedback["humor_preference"]
                character_config["writing_style"]["humor_style"] = humor_style
            
            if "response_length_preference" in user_feedback:
                # 调整回复长度偏好
                length_pref = user_feedback["response_length_preference"]
                if length_pref == "shorter":
                    character_config["writing_style"]["sentence_length"] = "短句"
                elif length_pref == "longer":
                    character_config["writing_style"]["sentence_length"] = "长句"
            
            # 记录调整
            logger.info(f"根据用户反馈调整角色 {character_id}")
            return True
            
        except Exception as e:
            logger.error(f"调整角色失败: {e}")
            return False
    
    def get_character_stats(self, character_id: str) -> Dict[str, Any]:
        """获取角色统计信息"""
        try:
            if character_id not in self.character_profiles:
                return {}
            
            character_config = self.character_profiles[character_id]
            
            return {
                "character_id": character_id,
                "name": character_config["name"],
                "personality_scores": character_config["personality"],
                "knowledge_domains": character_config["knowledge_domains"],
                "consistency_checks": self.consistency_cache.get(character_id, 0),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取角色统计失败: {e}")
            return {}
