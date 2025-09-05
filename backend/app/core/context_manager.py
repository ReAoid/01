"""
上下文管理系统

实现智能上下文管理、裁剪算法和会话摘要功能
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

import numpy as np
from transformers import AutoTokenizer

from ..config import get_settings
from ..models.chat_models import ChatMessage, MessageRole, ConversationContext
from ..models.memory_models import ConversationSummary
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ContextManager:
    """上下文管理器"""
    
    def __init__(self):
        self.tokenizer = None
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """初始化分词器"""
        try:
            # 使用通用的中文分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-chinese",
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"无法加载分词器: {e}，使用简单估算")
            self.tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"分词失败: {e}，使用简单估算")
        
        # 简单估算：中文按字符数，英文按单词数的1.3倍
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        other_chars = len(text) - chinese_chars - english_words
        
        return int(chinese_chars + english_words * 1.3 + other_chars * 0.5)
    
    def calculate_message_importance(self, message: ChatMessage) -> float:
        """计算消息重要性评分"""
        score = 0.0
        content = message.content.lower()
        
        # 基础评分
        if message.role == MessageRole.USER:
            score += 1.0  # 用户消息基础分
            
            # 问题类消息权重更高
            if any(marker in content for marker in ["?", "？", "请", "怎么", "如何", "为什么"]):
                score += 2.0
            
            # 命令或请求
            if any(marker in content for marker in ["帮我", "请帮", "能否", "可以", "需要"]):
                score += 1.5
        
        elif message.role == MessageRole.ASSISTANT:
            score += 0.8  # AI回复基础分
            
            # 包含详细信息的回复
            if len(content) > 200:
                score += 1.0
            
            # 包含代码或结构化信息
            if any(marker in content for marker in ["```", "1.", "2.", "•", "·"]):
                score += 1.2
        
        # 情感强度加权
        emotion_keywords = {
            "高兴": 1.2, "开心": 1.2, "兴奋": 1.3,
            "难过": 1.5, "伤心": 1.5, "沮丧": 1.4,
            "愤怒": 1.6, "生气": 1.6, "恼火": 1.4,
            "担心": 1.3, "焦虑": 1.4, "紧张": 1.3,
            "感谢": 1.1, "谢谢": 1.1, "抱歉": 1.2,
        }
        
        for keyword, multiplier in emotion_keywords.items():
            if keyword in content:
                score *= multiplier
                break
        
        # 信息密度评分
        info_density = self._calculate_info_density(content)
        score += info_density
        
        # 时间衰减（最近的消息权重稍高）
        time_diff = datetime.utcnow() - message.created_at
        if time_diff.total_seconds() < 3600:  # 1小时内
            score *= 1.1
        elif time_diff.total_seconds() < 86400:  # 1天内
            score *= 1.05
        
        return min(score, 10.0)  # 限制最大评分
    
    def _calculate_info_density(self, text: str) -> float:
        """计算信息密度"""
        if not text:
            return 0.0
        
        # 实体识别（简单版本）
        entities = 0
        
        # 人名、地名、组织名等
        entities += len(re.findall(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|博士|教授)', text))
        entities += len(re.findall(r'[\u4e00-\u9fff]{2,6}(?:公司|学校|大学|医院|银行)', text))
        entities += len(re.findall(r'[\u4e00-\u9fff]{2,8}(?:市|省|县|区|街|路)', text))
        
        # 数字、日期、时间
        entities += len(re.findall(r'\d+', text))
        entities += len(re.findall(r'\d{4}年|\d{1,2}月|\d{1,2}日', text))
        
        # 专业术语（技术词汇）
        tech_keywords = [
            'api', 'sdk', 'ai', 'ml', '机器学习', '人工智能', '算法',
            '数据库', '服务器', '网络', '系统', '框架', '开发', '编程'
        ]
        entities += sum(1 for keyword in tech_keywords if keyword in text.lower())
        
        # 归一化信息密度
        text_length = len(text)
        density = entities / max(text_length / 50, 1)  # 每50字符的实体密度
        
        return min(density * 2, 3.0)  # 限制最大密度分数
    
    async def smart_context_trimming(
        self,
        messages: List[ChatMessage],
        max_tokens: int = None
    ) -> Tuple[List[ChatMessage], Optional[str]]:
        """智能上下文裁剪算法"""
        if not messages:
            return [], None
        
        max_tokens = max_tokens or settings.system.max_context_tokens
        
        # 1. 计算每条消息的重要性评分
        scored_messages = []
        for msg in messages:
            score = self.calculate_message_importance(msg)
            tokens = self.estimate_tokens(msg.content)
            scored_messages.append((score, tokens, msg))
        
        # 2. 计算总token数
        total_tokens = sum(tokens for _, tokens, _ in scored_messages)
        
        if total_tokens <= max_tokens:
            return messages, None
        
        # 3. 保留最近的消息（确保对话连贯性）
        recent_count = min(5, len(messages))
        recent_messages = messages[-recent_count:]
        recent_tokens = sum(self.estimate_tokens(msg.content) for msg in recent_messages)
        
        # 4. 为历史消息分配剩余token空间
        remaining_tokens = max_tokens - recent_tokens
        if remaining_tokens <= 0:
            # 如果最近消息就超过限制，需要生成摘要
            summary = await self._generate_conversation_summary(messages[:-recent_count])
            return recent_messages, summary
        
        # 5. 按重要性选择历史消息
        historical_messages = scored_messages[:-recent_count]
        historical_messages.sort(key=lambda x: x[0], reverse=True)  # 按重要性排序
        
        selected_historical = []
        used_tokens = 0
        
        for score, tokens, msg in historical_messages:
            if used_tokens + tokens <= remaining_tokens:
                selected_historical.append(msg)
                used_tokens += tokens
            else:
                break
        
        # 6. 按时间顺序重新排列
        selected_historical.sort(key=lambda x: x.created_at)
        final_messages = selected_historical + recent_messages
        
        # 7. 如果还有很多消息被删除，生成摘要
        summary = None
        removed_count = len(messages) - len(final_messages)
        if removed_count > 10:  # 如果删除了超过10条消息
            removed_messages = [msg for msg in messages if msg not in final_messages]
            summary = await self._generate_conversation_summary(removed_messages)
        
        logger.info(f"上下文裁剪完成: {len(messages)} -> {len(final_messages)} 条消息")
        return final_messages, summary
    
    async def _generate_conversation_summary(
        self,
        messages: List[ChatMessage]
    ) -> str:
        """生成对话摘要"""
        if not messages:
            return ""
        
        try:
            # 提取关键信息
            key_points = []
            user_questions = []
            ai_responses = []
            
            for msg in messages:
                if msg.role == MessageRole.USER:
                    if len(msg.content) > 20:  # 过滤太短的消息
                        if any(marker in msg.content for marker in ["?", "？", "请", "怎么"]):
                            user_questions.append(msg.content[:100])
                        else:
                            key_points.append(f"用户: {msg.content[:100]}")
                
                elif msg.role == MessageRole.ASSISTANT:
                    if len(msg.content) > 50:
                        ai_responses.append(msg.content[:150])
            
            # 构建摘要
            summary_parts = []
            
            if user_questions:
                summary_parts.append(f"主要问题: {'; '.join(user_questions[:3])}")
            
            if key_points:
                summary_parts.append(f"关键信息: {'; '.join(key_points[:3])}")
            
            if ai_responses:
                summary_parts.append(f"主要回复: {'; '.join(ai_responses[:2])}")
            
            # 时间范围
            if len(messages) > 1:
                start_time = messages[0].created_at.strftime("%H:%M")
                end_time = messages[-1].created_at.strftime("%H:%M")
                summary_parts.append(f"时间: {start_time}-{end_time}")
            
            summary = " | ".join(summary_parts)
            return summary[:500]  # 限制摘要长度
            
        except Exception as e:
            logger.error(f"生成对话摘要失败: {e}")
            return f"对话摘要 ({len(messages)}条消息)"
    
    async def build_context(
        self,
        session_id: UUID,
        user_input: str,
        messages: List[ChatMessage],
        character_config: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None
    ) -> ConversationContext:
        """构建完整的对话上下文"""
        
        # 1. 智能裁剪上下文
        trimmed_messages, summary = await self.smart_context_trimming(messages)
        
        # 2. 计算token使用量
        total_tokens = sum(self.estimate_tokens(msg.content) for msg in trimmed_messages)
        total_tokens += self.estimate_tokens(user_input)
        
        # 3. 添加摘要到上下文（如果有）
        if summary:
            total_tokens += self.estimate_tokens(summary)
        
        # 4. 提取关键实体和主题
        key_entities = self._extract_key_entities(trimmed_messages)
        emotion_history = self._extract_emotion_history(trimmed_messages)
        
        # 5. 构建上下文对象
        context = ConversationContext(
            session_id=session_id,
            messages=trimmed_messages,
            total_tokens=total_tokens,
            summary=summary,
            key_entities=key_entities,
            emotion_history=emotion_history
        )
        
        logger.info(f"构建上下文完成: {len(trimmed_messages)}条消息, {total_tokens} tokens")
        return context
    
    def _extract_key_entities(self, messages: List[ChatMessage]) -> List[str]:
        """提取关键实体"""
        entities = set()
        
        for msg in messages:
            content = msg.content
            
            # 简单的实体提取
            # 人名
            person_matches = re.findall(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|博士|教授)', content)
            entities.update(person_matches)
            
            # 地名
            place_matches = re.findall(r'[\u4e00-\u9fff]{2,8}(?:市|省|县|区|街|路)', content)
            entities.update(place_matches)
            
            # 组织机构
            org_matches = re.findall(r'[\u4e00-\u9fff]{2,10}(?:公司|学校|大学|医院|银行)', content)
            entities.update(org_matches)
            
            # 产品/服务
            product_matches = re.findall(r'[\u4e00-\u9fff]{2,8}(?:系统|平台|软件|应用|服务)', content)
            entities.update(product_matches)
        
        return list(entities)[:20]  # 限制数量
    
    def _extract_emotion_history(self, messages: List[ChatMessage]) -> List[str]:
        """提取情感历程"""
        emotions = []
        
        emotion_keywords = {
            "开心": ["开心", "高兴", "兴奋", "愉快", "满意", "棒", "好"],
            "难过": ["难过", "伤心", "沮丧", "失望", "痛苦"],
            "愤怒": ["愤怒", "生气", "恼火", "气愤", "烦躁"],
            "担心": ["担心", "焦虑", "紧张", "害怕", "忧虑"],
            "感谢": ["感谢", "谢谢", "感激", "感恩"],
            "抱歉": ["抱歉", "对不起", "不好意思", "sorry"]
        }
        
        for msg in messages[-10:]:  # 只分析最近10条消息
            content = msg.content.lower()
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in content for keyword in keywords):
                    emotions.append(emotion)
                    break
        
        return emotions
    
    async def should_generate_summary(
        self,
        session_id: UUID,
        current_tokens: int,
        message_count: int
    ) -> bool:
        """判断是否应该生成会话摘要"""
        
        # 基于token数量
        if current_tokens >= settings.ai.context.summary_trigger_tokens:
            return True
        
        # 基于消息数量
        if message_count >= settings.ai.context.max_history_messages:
            return True
        
        # 基于时间（长时间对话）
        # 这里可以添加基于会话持续时间的逻辑
        
        return False
    
    def format_context_for_ai(
        self,
        context: ConversationContext,
        character_config: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None,
        user_input: str = ""
    ) -> List[Dict[str, str]]:
        """将上下文格式化为AI模型可用的格式"""
        
        formatted_messages = []
        
        # 1. 系统指令
        system_prompt = self._build_system_prompt(character_config, memory_context)
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 2. 会话摘要（如果有）
        if context.summary:
            formatted_messages.append({
                "role": "system",
                "content": f"对话历史摘要: {context.summary}"
            })
        
        # 3. 历史消息
        for msg in context.messages:
            formatted_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # 4. 当前用户输入
        if user_input:
            formatted_messages.append({
                "role": "user",
                "content": user_input
            })
        
        return formatted_messages
    
    def _build_system_prompt(
        self,
        character_config: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None
    ) -> str:
        """构建系统提示词"""
        
        prompt_parts = []
        
        # 基础角色设定
        if character_config:
            name = character_config.get("name", "AI助手")
            personality = character_config.get("personality", {})
            
            prompt_parts.append(f"你是{name}，一个智能AI助手。")
            
            # 性格特征
            if personality.get("traits"):
                traits = ", ".join(personality["traits"])
                prompt_parts.append(f"你的性格特点是: {traits}。")
            
            # 表达风格
            writing_style = character_config.get("writing_style", {})
            if writing_style:
                style_desc = []
                if writing_style.get("tone"):
                    style_desc.append(f"语调{writing_style['tone']}")
                if writing_style.get("use_emoji"):
                    style_desc.append("适当使用表情符号")
                if style_desc:
                    prompt_parts.append(f"表达风格: {', '.join(style_desc)}。")
        
        # 记忆上下文
        if memory_context:
            prompt_parts.append(f"相关记忆: {memory_context}")
        
        # 基本指导原则
        prompt_parts.extend([
            "请根据对话历史和用户需求提供有帮助的回复。",
            "保持回复的连贯性和一致性。",
            "如果不确定答案，请诚实说明。"
        ])
        
        return "\n".join(prompt_parts)
