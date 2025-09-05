"""
长期记忆管理系统

实现用户记忆的存储、检索、更新和分析功能
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..models.chat_models import ChatMessage, MessageRole
from ..models.memory_models import (
    Memory, MemoryType, MemoryImportance, Entity, Relationship,
    ConversationSummary, UserMemoryProfile, MemoryInsight
)
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self):
        self.embedding_model = None
        self.memory_cache = {}  # 简单的内存缓存
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """初始化向量化模型"""
        try:
            # 使用中文向量化模型
            self.embedding_model = SentenceTransformer(
                'shibing624/text2vec-base-chinese',
                device='cpu'
            )
            logger.info("向量化模型加载成功")
        except Exception as e:
            logger.warning(f"无法加载向量化模型: {e}")
            self.embedding_model = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本向量"""
        if not self.embedding_model or not text.strip():
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            return None
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算向量相似度"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    async def extract_memories_from_conversation(
        self,
        user_id: UUID,
        session_id: UUID,
        messages: List[ChatMessage]
    ) -> List[Memory]:
        """从对话中提取记忆"""
        memories = []
        
        try:
            # 分析对话内容
            conversation_text = "\n".join([
                f"{msg.role.value}: {msg.content}" 
                for msg in messages[-10:]  # 分析最近10条消息
            ])
            
            # 提取不同类型的记忆
            fact_memories = await self._extract_fact_memories(user_id, session_id, messages)
            preference_memories = await self._extract_preference_memories(user_id, session_id, messages)
            relationship_memories = await self._extract_relationship_memories(user_id, session_id, messages)
            event_memories = await self._extract_event_memories(user_id, session_id, messages)
            
            memories.extend(fact_memories)
            memories.extend(preference_memories)
            memories.extend(relationship_memories)
            memories.extend(event_memories)
            
            logger.info(f"从对话中提取了 {len(memories)} 条记忆")
            return memories
            
        except Exception as e:
            logger.error(f"提取记忆失败: {e}")
            return []
    
    async def _extract_fact_memories(
        self,
        user_id: UUID,
        session_id: UUID,
        messages: List[ChatMessage]
    ) -> List[Memory]:
        """提取事实性记忆"""
        memories = []
        
        # 事实性陈述的模式
        fact_patterns = [
            r'我是(.{2,20})',
            r'我在(.{2,20})工作',
            r'我住在(.{2,20})',
            r'我的(.{2,10})是(.{2,20})',
            r'我有(.{2,20})',
            r'我会(.{2,20})',
            r'我学过(.{2,20})',
        ]
        
        for msg in messages:
            if msg.role != MessageRole.USER:
                continue
                
            content = msg.content
            
            for pattern in fact_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        fact_content = f"用户{pattern.replace('(.{2,20})', match[0] if len(match) == 1 else f'{match[0]}是{match[1]}')}"
                    else:
                        fact_content = f"用户{pattern.replace('(.{2,20})', match)}"
                    
                    memory = Memory(
                        user_id=user_id,
                        session_id=session_id,
                        content=fact_content,
                        memory_type=MemoryType.FACT,
                        importance=MemoryImportance.HIGH,
                        source="conversation_extraction",
                        embedding=self.generate_embedding(fact_content)
                    )
                    memories.append(memory)
        
        return memories
    
    async def _extract_preference_memories(
        self,
        user_id: UUID,
        session_id: UUID,
        messages: List[ChatMessage]
    ) -> List[Memory]:
        """提取偏好记忆"""
        memories = []
        
        # 偏好表达的模式
        preference_patterns = [
            r'我喜欢(.{2,20})',
            r'我不喜欢(.{2,20})',
            r'我讨厌(.{2,20})',
            r'我爱(.{2,20})',
            r'我偏好(.{2,20})',
            r'我倾向于(.{2,20})',
            r'我更愿意(.{2,20})',
        ]
        
        for msg in messages:
            if msg.role != MessageRole.USER:
                continue
                
            content = msg.content
            
            for pattern in preference_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    preference_content = f"用户{pattern.replace('(.{2,20})', match)}"
                    
                    # 判断偏好强度
                    importance = MemoryImportance.MEDIUM
                    if any(word in content for word in ["非常", "特别", "极其", "最"]):
                        importance = MemoryImportance.HIGH
                    elif any(word in content for word in ["有点", "稍微", "还行"]):
                        importance = MemoryImportance.LOW
                    
                    memory = Memory(
                        user_id=user_id,
                        session_id=session_id,
                        content=preference_content,
                        memory_type=MemoryType.PREFERENCE,
                        importance=importance,
                        source="conversation_extraction",
                        embedding=self.generate_embedding(preference_content)
                    )
                    memories.append(memory)
        
        return memories
    
    async def _extract_relationship_memories(
        self,
        user_id: UUID,
        session_id: UUID,
        messages: List[ChatMessage]
    ) -> List[Memory]:
        """提取关系记忆"""
        memories = []
        
        # 关系描述的模式
        relationship_patterns = [
            r'我的(.{2,10})(.{2,10})',
            r'(.{2,10})是我的(.{2,10})',
            r'我和(.{2,10})(.{2,20})',
        ]
        
        for msg in messages:
            if msg.role != MessageRole.USER:
                continue
                
            content = msg.content
            
            for pattern in relationship_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        relationship_content = f"用户与{match[0]}的关系: {match[1]}"
                        
                        memory = Memory(
                            user_id=user_id,
                            session_id=session_id,
                            content=relationship_content,
                            memory_type=MemoryType.RELATIONSHIP,
                            importance=MemoryImportance.MEDIUM,
                            source="conversation_extraction",
                            embedding=self.generate_embedding(relationship_content)
                        )
                        memories.append(memory)
        
        return memories
    
    async def _extract_event_memories(
        self,
        user_id: UUID,
        session_id: UUID,
        messages: List[ChatMessage]
    ) -> List[Memory]:
        """提取事件记忆"""
        memories = []
        
        # 事件描述的模式
        event_patterns = [
            r'昨天(.{5,50})',
            r'今天(.{5,50})',
            r'明天(.{5,50})',
            r'上周(.{5,50})',
            r'下周(.{5,50})',
            r'最近(.{5,50})',
            r'刚才(.{5,50})',
        ]
        
        for msg in messages:
            if msg.role != MessageRole.USER:
                continue
                
            content = msg.content
            
            for pattern in event_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    event_content = f"用户事件: {pattern.split('(')[0]}{match}"
                    
                    memory = Memory(
                        user_id=user_id,
                        session_id=session_id,
                        content=event_content,
                        memory_type=MemoryType.EVENT,
                        importance=MemoryImportance.MEDIUM,
                        source="conversation_extraction",
                        embedding=self.generate_embedding(event_content),
                        # 事件记忆有时效性
                        expires_at=datetime.utcnow() + timedelta(days=30)
                    )
                    memories.append(memory)
        
        return memories
    
    async def retrieve_relevant_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Memory, float]]:
        """检索相关记忆"""
        try:
            # 生成查询向量
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # 这里应该从数据库查询用户的所有记忆
            # 为了演示，我们使用缓存中的记忆
            user_memories = self.memory_cache.get(str(user_id), [])
            
            relevant_memories = []
            
            for memory in user_memories:
                if memory.embedding:
                    similarity = self.calculate_similarity(query_embedding, memory.embedding)
                    
                    if similarity >= similarity_threshold:
                        relevant_memories.append((memory, similarity))
            
            # 按相似度排序
            relevant_memories.sort(key=lambda x: x[1], reverse=True)
            
            # 应用时间衰减和重要性权重
            weighted_memories = []
            for memory, similarity in relevant_memories[:limit * 2]:
                # 时间衰减因子
                time_diff = datetime.utcnow() - memory.created_at
                time_decay = max(0.5, 1.0 - (time_diff.total_seconds() / (30 * 24 * 3600)))  # 30天衰减
                
                # 重要性权重
                importance_weight = {
                    MemoryImportance.LOW: 0.7,
                    MemoryImportance.MEDIUM: 1.0,
                    MemoryImportance.HIGH: 1.3,
                    MemoryImportance.CRITICAL: 1.5
                }.get(memory.importance, 1.0)
                
                # 访问频次权重
                access_weight = min(1.5, 1.0 + memory.access_count * 0.1)
                
                # 综合评分
                final_score = similarity * time_decay * importance_weight * access_weight
                weighted_memories.append((memory, final_score))
            
            # 重新排序并返回
            weighted_memories.sort(key=lambda x: x[1], reverse=True)
            return weighted_memories[:limit]
            
        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return []
    
    async def update_memory_access(self, memory_id: UUID):
        """更新记忆访问记录"""
        try:
            # 这里应该更新数据库中的记忆访问信息
            # 为了演示，我们更新缓存
            for user_memories in self.memory_cache.values():
                for memory in user_memories:
                    if memory.id == memory_id:
                        memory.access_count += 1
                        memory.accessed_at = datetime.utcnow()
                        break
        except Exception as e:
            logger.error(f"更新记忆访问记录失败: {e}")
    
    async def generate_conversation_summary(
        self,
        user_id: UUID,
        session_id: UUID,
        messages: List[ChatMessage]
    ) -> ConversationSummary:
        """生成对话总结"""
        try:
            if not messages:
                raise ValueError("消息列表为空")
            
            # 基本统计
            message_count = len(messages)
            total_tokens = sum(len(msg.content.split()) for msg in messages)
            
            # 时间范围
            conversation_start = min(msg.created_at for msg in messages)
            conversation_end = max(msg.created_at for msg in messages)
            duration = (conversation_end - conversation_start).total_seconds() / 60
            
            # 提取关键点
            key_points = self._extract_key_points(messages)
            main_topics = self._extract_main_topics(messages)
            
            # 提取实体和关系
            entities = self._extract_entities_from_messages(messages)
            relationships = self._extract_relationships_from_messages(messages)
            
            # 情感分析
            sentiment = self._analyze_overall_sentiment(messages)
            emotion_journey = self._analyze_emotion_journey(messages)
            
            # 生成摘要文本
            summary_text = self._generate_summary_text(messages, key_points, main_topics)
            
            # 创建摘要对象
            summary = ConversationSummary(
                session_id=session_id,
                user_id=user_id,
                summary=summary_text,
                key_points=key_points,
                main_topics=main_topics,
                entities_mentioned=entities,
                relationships_formed=relationships,
                overall_sentiment=sentiment,
                emotion_journey=emotion_journey,
                message_count=message_count,
                token_count=total_tokens,
                duration_minutes=duration,
                conversation_start=conversation_start,
                conversation_end=conversation_end
            )
            
            logger.info(f"生成对话总结完成: {message_count}条消息, {len(key_points)}个关键点")
            return summary
            
        except Exception as e:
            logger.error(f"生成对话总结失败: {e}")
            raise
    
    def _extract_key_points(self, messages: List[ChatMessage]) -> List[str]:
        """提取关键点"""
        key_points = []
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                # 用户的重要陈述
                if any(marker in msg.content for marker in ["我要", "我需要", "请帮我", "问题是"]):
                    key_points.append(msg.content[:100])
            elif msg.role == MessageRole.ASSISTANT:
                # AI的重要回复
                if len(msg.content) > 100 and any(marker in msg.content for marker in ["总结", "建议", "方案", "步骤"]):
                    key_points.append(msg.content[:150])
        
        return key_points[:5]  # 限制数量
    
    def _extract_main_topics(self, messages: List[ChatMessage]) -> List[str]:
        """提取主要话题"""
        topics = set()
        
        # 简单的话题提取
        topic_keywords = {
            "技术": ["编程", "开发", "代码", "算法", "系统", "软件", "网站", "app"],
            "工作": ["工作", "职业", "公司", "项目", "任务", "会议", "同事"],
            "学习": ["学习", "课程", "考试", "书籍", "知识", "技能", "培训"],
            "生活": ["生活", "家庭", "朋友", "健康", "运动", "旅行", "美食"],
            "娱乐": ["电影", "音乐", "游戏", "小说", "动漫", "综艺", "体育"],
            "购物": ["购买", "商品", "价格", "品牌", "质量", "推荐", "比较"],
        }
        
        conversation_text = " ".join([msg.content for msg in messages]).lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in conversation_text for keyword in keywords):
                topics.add(topic)
        
        return list(topics)
    
    def _extract_entities_from_messages(self, messages: List[ChatMessage]) -> List[Entity]:
        """从消息中提取实体"""
        entities = []
        
        # 简化的实体提取
        for msg in messages:
            content = msg.content
            
            # 人名
            person_matches = re.findall(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|博士|教授)', content)
            for match in person_matches:
                entities.append(Entity(
                    name=match,
                    entity_type=EntityType.PERSON,
                    confidence=0.8
                ))
            
            # 地名
            place_matches = re.findall(r'[\u4e00-\u9fff]{2,8}(?:市|省|县|区|街|路)', content)
            for match in place_matches:
                entities.append(Entity(
                    name=match,
                    entity_type=EntityType.PLACE,
                    confidence=0.7
                ))
        
        return entities[:10]  # 限制数量
    
    def _extract_relationships_from_messages(self, messages: List[ChatMessage]) -> List[Relationship]:
        """从消息中提取关系"""
        relationships = []
        
        # 简化的关系提取
        for msg in messages:
            if msg.role != MessageRole.USER:
                continue
                
            content = msg.content
            
            # 简单的关系模式
            relationship_patterns = [
                (r'我的(.+?)是(.+)', "拥有"),
                (r'(.+?)是我的(.+)', "属于"),
                (r'我和(.+?)(.+)', "关系"),
            ]
            
            for pattern, relation_type in relationship_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        relationships.append(Relationship(
                            subject="用户",
                            predicate=relation_type,
                            object=match[0],
                            confidence=0.6,
                            context=content[:50]
                        ))
        
        return relationships[:5]  # 限制数量
    
    def _analyze_overall_sentiment(self, messages: List[ChatMessage]) -> str:
        """分析整体情感"""
        positive_words = ["好", "棒", "喜欢", "满意", "开心", "高兴", "谢谢", "感谢"]
        negative_words = ["不好", "糟糕", "讨厌", "失望", "难过", "生气", "抱歉", "问题"]
        
        positive_count = 0
        negative_count = 0
        
        for msg in messages:
            content = msg.content.lower()
            positive_count += sum(1 for word in positive_words if word in content)
            negative_count += sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_emotion_journey(self, messages: List[ChatMessage]) -> List[str]:
        """分析情感历程"""
        emotions = []
        
        emotion_keywords = {
            "开心": ["开心", "高兴", "兴奋", "愉快"],
            "难过": ["难过", "伤心", "沮丧", "失望"],
            "愤怒": ["愤怒", "生气", "恼火", "气愤"],
            "担心": ["担心", "焦虑", "紧张", "害怕"],
            "感谢": ["感谢", "谢谢", "感激"],
        }
        
        for msg in messages:
            content = msg.content.lower()
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in content for keyword in keywords):
                    emotions.append(emotion)
                    break
        
        return emotions
    
    def _generate_summary_text(
        self,
        messages: List[ChatMessage],
        key_points: List[str],
        main_topics: List[str]
    ) -> str:
        """生成摘要文本"""
        try:
            # 统计信息
            user_message_count = sum(1 for msg in messages if msg.role == MessageRole.USER)
            ai_message_count = sum(1 for msg in messages if msg.role == MessageRole.ASSISTANT)
            
            # 构建摘要
            summary_parts = []
            
            if main_topics:
                summary_parts.append(f"主要讨论了{', '.join(main_topics)}相关话题")
            
            summary_parts.append(f"用户发送了{user_message_count}条消息，AI回复了{ai_message_count}条")
            
            if key_points:
                summary_parts.append(f"关键内容包括: {'; '.join(key_points[:2])}")
            
            return "。".join(summary_parts) + "。"
            
        except Exception as e:
            logger.error(f"生成摘要文本失败: {e}")
            return f"对话包含{len(messages)}条消息"
    
    async def store_memory(self, memory: Memory) -> bool:
        """存储记忆"""
        try:
            # 这里应该存储到数据库
            # 为了演示，我们存储到内存缓存
            user_id_str = str(memory.user_id)
            if user_id_str not in self.memory_cache:
                self.memory_cache[user_id_str] = []
            
            self.memory_cache[user_id_str].append(memory)
            
            logger.info(f"存储记忆成功: {memory.memory_type.value} - {memory.content[:50]}")
            return True
            
        except Exception as e:
            logger.error(f"存储记忆失败: {e}")
            return False
    
    async def get_user_memory_profile(self, user_id: UUID) -> UserMemoryProfile:
        """获取用户记忆档案"""
        try:
            user_memories = self.memory_cache.get(str(user_id), [])
            
            # 统计信息
            total_memories = len(user_memories)
            memories_by_type = {}
            memories_by_importance = {}
            
            for memory in user_memories:
                # 按类型统计
                mem_type = memory.memory_type
                memories_by_type[mem_type] = memories_by_type.get(mem_type, 0) + 1
                
                # 按重要性统计
                importance = memory.importance
                memories_by_importance[importance] = memories_by_importance.get(importance, 0) + 1
            
            # 提取关键实体和关系
            all_entities = []
            all_relationships = []
            
            for memory in user_memories:
                if memory.entities:
                    all_entities.extend(memory.entities)
                if memory.relationships:
                    all_relationships.extend(memory.relationships)
            
            # 创建用户记忆档案
            profile = UserMemoryProfile(
                user_id=user_id,
                total_memories=total_memories,
                memories_by_type=memories_by_type,
                memories_by_importance=memories_by_importance,
                top_entities=all_entities[:10],
                key_relationships=all_relationships[:5]
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"获取用户记忆档案失败: {e}")
            return UserMemoryProfile(user_id=user_id)
    
    async def cleanup_expired_memories(self):
        """清理过期记忆"""
        try:
            current_time = datetime.utcnow()
            cleaned_count = 0
            
            for user_id, memories in self.memory_cache.items():
                # 过滤掉过期的记忆
                valid_memories = []
                for memory in memories:
                    if not memory.expires_at or memory.expires_at > current_time:
                        valid_memories.append(memory)
                    else:
                        cleaned_count += 1
                
                self.memory_cache[user_id] = valid_memories
            
            if cleaned_count > 0:
                logger.info(f"清理了{cleaned_count}条过期记忆")
                
        except Exception as e:
            logger.error(f"清理过期记忆失败: {e}")
