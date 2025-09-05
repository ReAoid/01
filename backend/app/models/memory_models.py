"""
记忆系统相关数据模型

定义长期记忆、对话总结等相关的Pydantic模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class MemoryType(str, Enum):
    """记忆类型枚举"""
    FACT = "fact"  # 事实信息
    PREFERENCE = "preference"  # 用户偏好
    RELATIONSHIP = "relationship"  # 关系信息
    EVENT = "event"  # 事件记录
    EMOTION = "emotion"  # 情感记录
    SKILL = "skill"  # 技能/能力
    GOAL = "goal"  # 目标/计划
    CONTEXT = "context"  # 上下文信息


class MemoryImportance(str, Enum):
    """记忆重要性枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EntityType(str, Enum):
    """实体类型枚举"""
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    EVENT = "event"
    CONCEPT = "concept"
    PRODUCT = "product"
    DATE = "date"
    NUMBER = "number"


class Entity(BaseModel):
    """实体模型"""
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    aliases: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class Relationship(BaseModel):
    """关系模型"""
    subject: str  # 主体
    predicate: str  # 谓语/关系类型
    object: str  # 客体
    confidence: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None
    
    class Config:
        from_attributes = True


class Memory(BaseModel):
    """记忆模型"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: Optional[UUID] = None  # 关联的会话ID
    
    # 记忆内容
    content: str = Field(..., min_length=1, max_length=2000)
    memory_type: MemoryType
    importance: MemoryImportance = MemoryImportance.MEDIUM
    
    # 提取的信息
    entities: Optional[List[Entity]] = None
    relationships: Optional[List[Relationship]] = None
    keywords: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    
    # 元数据
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: str = "conversation"  # conversation, user_input, system
    context: Optional[str] = None
    
    # 向量化信息
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # 时间信息
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    
    # 关联信息
    related_memories: Optional[List[UUID]] = None
    tags: Optional[List[str]] = None
    
    # 生命周期
    expires_at: Optional[datetime] = None
    is_archived: bool = False
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class MemoryCreate(BaseModel):
    """创建记忆请求模型"""
    content: str = Field(..., min_length=1, max_length=2000)
    memory_type: MemoryType
    importance: MemoryImportance = MemoryImportance.MEDIUM
    session_id: Optional[UUID] = None
    context: Optional[str] = None
    tags: Optional[List[str]] = None
    
    @validator("content")
    def validate_content(cls, v):
        """验证记忆内容"""
        if not v.strip():
            raise ValueError("记忆内容不能为空")
        return v.strip()


class MemoryResponse(BaseModel):
    """记忆响应模型"""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID] = None
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    entities: Optional[List[Entity]] = None
    relationships: Optional[List[Relationship]] = None
    keywords: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    confidence: float
    source: str
    context: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int
    related_memories: Optional[List[UUID]] = None
    tags: Optional[List[str]] = None
    is_archived: bool
    
    class Config:
        from_attributes = True


class MemorySearch(BaseModel):
    """记忆搜索请求模型"""
    query: str = Field(..., min_length=1, max_length=500)
    memory_types: Optional[List[MemoryType]] = None
    importance_levels: Optional[List[MemoryImportance]] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=10, ge=1, le=100)
    include_archived: bool = False


class MemorySearchResult(BaseModel):
    """记忆搜索结果"""
    memory: MemoryResponse
    similarity_score: float = Field(ge=0.0, le=1.0)
    relevance_reason: Optional[str] = None
    
    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    """对话总结模型"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID
    
    # 总结内容
    summary: str = Field(..., min_length=10, max_length=1000)
    key_points: List[str] = []
    main_topics: List[str] = []
    
    # 提取的信息
    entities_mentioned: Optional[List[Entity]] = None
    relationships_formed: Optional[List[Relationship]] = None
    user_preferences_learned: Optional[Dict[str, Any]] = None
    
    # 情感分析
    overall_sentiment: Optional[str] = None  # positive, negative, neutral
    emotion_journey: Optional[List[str]] = None
    
    # 统计信息
    message_count: int
    token_count: int
    duration_minutes: Optional[float] = None
    
    # 时间范围
    conversation_start: datetime
    conversation_end: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # 质量评估
    summary_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SummaryCreate(BaseModel):
    """创建对话总结请求模型"""
    session_id: UUID
    summary: str = Field(..., min_length=10, max_length=1000)
    key_points: List[str] = []
    main_topics: List[str] = []
    message_count: int = Field(ge=1)
    token_count: int = Field(ge=1)
    conversation_start: datetime
    conversation_end: datetime


class UserMemoryProfile(BaseModel):
    """用户记忆档案"""
    user_id: UUID
    
    # 统计信息
    total_memories: int = 0
    memories_by_type: Dict[MemoryType, int] = {}
    memories_by_importance: Dict[MemoryImportance, int] = {}
    
    # 实体和关系
    top_entities: List[Entity] = []
    key_relationships: List[Relationship] = []
    frequent_topics: List[str] = []
    
    # 偏好分析
    personality_traits: Optional[Dict[str, float]] = None
    interests: Optional[List[str]] = None
    communication_style: Optional[Dict[str, Any]] = None
    
    # 时间分析
    most_active_times: Optional[List[str]] = None
    memory_creation_pattern: Optional[Dict[str, int]] = None
    
    # 更新时间
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class MemoryCluster(BaseModel):
    """记忆聚类"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    cluster_name: str
    description: Optional[str] = None
    
    # 聚类信息
    memory_ids: List[UUID]
    centroid_embedding: Optional[List[float]] = None
    
    # 聚类特征
    common_entities: List[Entity] = []
    common_topics: List[str] = []
    dominant_memory_type: Optional[MemoryType] = None
    
    # 统计
    memory_count: int
    average_importance: float
    creation_date_range: Optional[Dict[str, datetime]] = None
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class MemoryInsight(BaseModel):
    """记忆洞察"""
    user_id: UUID
    insight_type: str  # pattern, preference, behavior, etc.
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    # 支持数据
    supporting_memories: List[UUID]
    evidence_count: int
    
    # 时间信息
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    relevance_score: float = Field(ge=0.0, le=1.0)
    
    # 可操作性
    actionable: bool = False
    suggested_actions: Optional[List[str]] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
