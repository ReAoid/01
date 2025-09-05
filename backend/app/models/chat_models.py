"""
聊天相关数据模型

定义聊天消息、会话等相关的Pydantic模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """消息类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    """消息状态枚举"""
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ERROR = "error"


class MediaInfo(BaseModel):
    """媒体信息模型"""
    file_name: str
    file_size: int
    mime_type: str
    url: str
    thumbnail_url: Optional[str] = None
    duration: Optional[float] = None  # 音频/视频时长（秒）
    dimensions: Optional[Dict[str, int]] = None  # 图片/视频尺寸 {"width": 1920, "height": 1080}
    ocr_text: Optional[str] = None  # OCR识别的文本


class MessageMetadata(BaseModel):
    """消息元数据"""
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    emotion_detected: Optional[str] = None
    confidence_score: Optional[float] = None
    character_used: Optional[str] = None
    memory_retrieved: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """聊天消息模型"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: MessageRole
    content: str
    message_type: MessageType = MessageType.TEXT
    status: MessageStatus = MessageStatus.SENT
    
    # 媒体相关
    media_info: Optional[MediaInfo] = None
    
    # 元数据
    metadata: Optional[MessageMetadata] = None
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # 关联信息
    parent_message_id: Optional[UUID] = None  # 回复的消息ID
    thread_id: Optional[UUID] = None  # 话题线程ID
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class MessageCreate(BaseModel):
    """创建消息请求模型"""
    content: str = Field(..., min_length=1, max_length=10000)
    message_type: MessageType = MessageType.TEXT
    media_info: Optional[MediaInfo] = None
    parent_message_id: Optional[UUID] = None
    
    @validator("content")
    def validate_content(cls, v):
        """验证消息内容"""
        if not v.strip():
            raise ValueError("消息内容不能为空")
        return v.strip()


class MessageResponse(BaseModel):
    """消息响应模型"""
    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    message_type: MessageType
    status: MessageStatus
    media_info: Optional[MediaInfo] = None
    metadata: Optional[MessageMetadata] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    parent_message_id: Optional[UUID] = None
    thread_id: Optional[UUID] = None
    
    class Config:
        from_attributes = True


class StreamChunk(BaseModel):
    """流式响应分块模型"""
    id: UUID
    session_id: UUID
    chunk_id: int  # 分块序号
    content: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SessionStatus(str, Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class SessionMetadata(BaseModel):
    """会话元数据"""
    total_messages: int = 0
    total_tokens: int = 0
    last_activity: Optional[datetime] = None
    character_name: Optional[str] = None
    session_summary: Optional[str] = None
    topics: Optional[List[str]] = None
    user_satisfaction: Optional[float] = None  # 用户满意度评分


class ChatSession(BaseModel):
    """聊天会话模型"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    title: str = "新对话"
    status: SessionStatus = SessionStatus.ACTIVE
    
    # 会话配置
    character_config: Optional[Dict[str, Any]] = None
    ai_model: Optional[str] = None
    temperature: Optional[float] = None
    
    # 元数据
    metadata: Optional[SessionMetadata] = None
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SessionCreate(BaseModel):
    """创建会话请求模型"""
    title: Optional[str] = "新对话"
    character_config: Optional[Dict[str, Any]] = None
    ai_model: Optional[str] = None
    temperature: Optional[float] = None
    
    @validator("title")
    def validate_title(cls, v):
        """验证会话标题"""
        if v and len(v.strip()) > 100:
            raise ValueError("会话标题不能超过100个字符")
        return v.strip() if v else "新对话"


class SessionResponse(BaseModel):
    """会话响应模型"""
    id: UUID
    user_id: UUID
    title: str
    status: SessionStatus
    character_config: Optional[Dict[str, Any]] = None
    ai_model: Optional[str] = None
    temperature: Optional[float] = None
    metadata: Optional[SessionMetadata] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # 可选包含最近消息
    recent_messages: Optional[List[MessageResponse]] = None
    
    class Config:
        from_attributes = True


class ConversationContext(BaseModel):
    """对话上下文模型"""
    session_id: UUID
    messages: List[ChatMessage]
    total_tokens: int
    summary: Optional[str] = None
    key_entities: Optional[List[str]] = None
    emotion_history: Optional[List[str]] = None
    
    class Config:
        from_attributes = True


class ChatStats(BaseModel):
    """聊天统计模型"""
    total_sessions: int
    total_messages: int
    total_tokens: int
    average_session_length: float
    most_active_hours: List[int]
    popular_topics: List[str]
    user_satisfaction_avg: Optional[float] = None
    
    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
